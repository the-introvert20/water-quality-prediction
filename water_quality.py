"""Prediction and validation helpers, independent of the Streamlit interface."""
import csv
import io
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from train_model import treat_zero_ph_as_missing

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
MODEL_NAMES = ['Random_Forest', 'SVC', 'Extra_Trees', 'Soft_Voting_RF_SVC',
               'XGBoost', 'Decision_Tree', 'Logistic_Regression']
CLASS_LABELS = {0: '0 = Not Potable', 1: '1 = Potable'}
CLASS_NAMES = {0: 'Not Potable', 1: 'Potable'}
DISCLAIMER = (
    'Educational ML estimate only. Predictions do not certify drinking-water safety. '
    'Laboratory testing and applicable water-quality standards are necessary to determine actual safety.'
)
# Preserve the original form limits. These are interface limits, not safety thresholds.
FIELD_SPECS = {
    'ph': ('pH', 14.0, 0.1, 7.0, 'Acidity / alkalinity; unitless. Zero is treated as missing.'),
    'Hardness': ('Hardness (mg/L)', 500.0, 1.0, 196.0, 'Calcium and magnesium concentration.'),
    'Solids': ('Dissolved solids (ppm)', 50000.0, 10.0, 20000.0, 'Total dissolved solids (TDS).'),
    'Chloramines': ('Chloramines (mg/L)', 20.0, 0.1, 7.0, 'Disinfectant-related measurement.'),
    'Sulfate': ('Sulfate (mg/L)', 500.0, 1.0, 333.0, 'Dissolved sulfate concentration.'),
    'Conductivity': ('Conductivity (µS/cm)', 1000.0, 1.0, 420.0, 'Electrical conductivity.'),
    'Organic_carbon': ('Organic carbon (mg/L)', 30.0, 0.1, 14.0, 'Total organic carbon.'),
    'Trihalomethanes': ('Trihalomethanes (µg/L)', 200.0, 0.1, 66.0, 'Disinfection byproducts.'),
    'Turbidity': ('Turbidity (NTU)', 10.0, 0.1, 4.0, 'Optical cloudiness.'),
}
RESULT_COLUMNS = ['Prediction', 'Predicted_Class', 'Max_Class_Probability',
                  'Prob_Not_Potable', 'Prob_Potable', 'Prediction_Model']


def load_pipeline(model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {model_name}')
    pipeline = joblib.load(BASE_DIR / 'models' / f'{model_name}.pkl')
    names = list(joblib.load(BASE_DIR / 'models' / 'feature_names.pkl'))
    if names != FEATURES:
        raise ValueError('Saved feature schema does not match the nine application inputs.')
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ['imputer', 'scaler', 'classifier']:
        raise ValueError('Expected a complete imputer / scaler / classifier Pipeline. Retrain the models.')
    check_is_fitted(pipeline)
    if list(pipeline.feature_names_in_) != FEATURES or set(pipeline.classes_) != {0, 1}:
        raise ValueError('Pipeline features or target classes do not match the application schema.')
    return pipeline


def decision_threshold(pipeline):
    """Return the Potable-probability threshold used by ``pipeline.predict`` (None = native rule)."""
    classifier = getattr(pipeline, 'named_steps', {}).get('classifier')
    threshold = getattr(classifier, 'threshold', None)
    return float(threshold) if threshold is not None else None


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept numeric or missing measurements, without imputing or mutating inputs."""
    if frame.empty:
        raise ValueError('The CSV has no data rows. Add at least one sample below the header.')
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique. Remove duplicate headers.')
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    numeric = pd.DataFrame(index=frame.index)
    for name in FEATURES:
        source = frame[name]
        blank = source.isna() | source.astype(str).str.strip().eq('')
        values = pd.Series(pd.to_numeric(source.mask(blank, np.nan), errors='coerce'), index=frame.index)
        invalid = (~blank & values.isna()) | np.isinf(values)
        if invalid.any():
            positions = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            raise ValueError(f'{name}: use finite numbers or blank cells. Invalid CSV row(s): {positions}.')
        numeric[name] = values.astype(float)
    return treat_zero_ph_as_missing(numeric)


def read_batch_csv(content):
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Save the CSV with UTF-8 encoding and upload it again.') from exc
    try:
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows)
        if len(header) != len(set(header)):
            raise ValueError('Column names must be unique. Remove duplicate headers.')
        for row_number, row in enumerate(rows, start=2):
            if row and len(row) != len(header):
                raise ValueError(f'CSV row {row_number} has {len(row)} fields; expected {len(header)}. Check comma separators.')
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (StopIteration, pd.errors.EmptyDataError) as exc:
        raise ValueError('The file is empty. Upload a CSV with a header and at least one sample.') from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise ValueError('Could not parse the CSV. Check comma separators and quoted fields.') from exc
    validate_features(frame)
    collisions = sorted(set(RESULT_COLUMNS).intersection(frame.columns))
    if collisions:
        raise ValueError('Rename existing result columns before analysis: ' + ', '.join(collisions))
    return frame


def predict_frame(pipeline, frame):
    prepared = validate_features(frame)
    predictions = np.asarray(pipeline.predict(prepared), dtype=int)
    probabilities = None
    if hasattr(pipeline, 'predict_proba'):
        raw = np.asarray(pipeline.predict_proba(prepared))
        # Always order by actual class identifiers, not an assumed output order.
        indices = [list(pipeline.classes_).index(label) for label in (0, 1)]
        probabilities = raw[:, indices]
    return predictions, probabilities, prepared


def batch_results(pipeline, frame, model_name):
    collisions = sorted(set(RESULT_COLUMNS).intersection(frame.columns))
    if collisions:
        raise ValueError('Rename existing result columns before analysis: ' + ', '.join(collisions))
    predictions, probabilities, prepared = predict_frame(pipeline, frame)
    result = frame.copy(deep=True)
    result['Prediction'] = [CLASS_NAMES[int(value)] for value in predictions]
    result['Predicted_Class'] = predictions
    if probabilities is not None:
        result['Max_Class_Probability'] = probabilities.max(axis=1) * 100
        result['Prob_Not_Potable'] = probabilities[:, 0] * 100
        result['Prob_Potable'] = probabilities[:, 1] * 100
    result['Prediction_Model'] = model_name
    return result, prepared


def feature_importance(pipeline):
    classifier = pipeline.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        values = classifier.feature_importances_
        label = 'Feature importance'
    elif hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
        label = 'Absolute standardized coefficient'
    else:
        return None, None
    return pd.DataFrame({'Feature': FEATURES, 'Importance': values}).sort_values('Importance'), label


def reference_checks(frame):
    """Retain the original app's reference rules without claiming regulatory validity."""
    sample = validate_features(frame).iloc[0]
    flags = []
    ph = sample['ph']
    if pd.notna(ph) and (ph < 6.5 or ph > 8.5):
        flags.append('pH is outside the project reference interval of 6.5–8.5.')
    for name, threshold, unit in [('Turbidity', 5, 'NTU'), ('Chloramines', 4, 'mg/L'), ('Sulfate', 250, 'mg/L')]:
        if pd.notna(sample[name]) and sample[name] > threshold:
            flags.append(f'{name} exceeds the project reference value of {threshold} {unit}.')
    return flags


def prediction_report(frame, prediction, probabilities, model_name, threshold=None):
    lines = ['WATER QUALITY INTELLIGENCE | PREDICTION REPORT',
             f'Generated: {datetime.now().isoformat(timespec="seconds")}',
             f'Model: {model_name}', f'Predicted class: {CLASS_LABELS[int(prediction)]}']
    if threshold is not None:
        lines.append(f'Decision rule: Potable only if P(Potable) >= {threshold:.2f} '
                     '(chosen from out-of-fold training predictions to limit false Potable predictions).')
    lines.extend(['', 'INPUT MEASUREMENTS (as submitted):'])
    for name in FEATURES:
        value = frame.iloc[0][name]
        lines.append(f'  {name}: {"missing" if pd.isna(value) else value}')
    if probabilities is not None:
        lines.extend(['', 'MODEL PROBABILITIES (not guaranteed real-world confidence):',
                      f'  {CLASS_LABELS[0]}: {probabilities[0]:.2%}',
                      f'  {CLASS_LABELS[1]}: {probabilities[1]:.2%}'])
    flags = reference_checks(frame)
    if flags:
        lines.extend(['', 'PROJECT REFERENCE CHECKS (legacy rules, not a standards-compliance assessment):', *flags])
    lines.extend(['', 'PREPROCESSING:',
                  'pH = 0 is treated as missing. Median imputation and scaling occur inside the fitted Pipeline.',
                  '', DISCLAIMER])
    return '\n'.join(lines)
