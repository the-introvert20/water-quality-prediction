"""Train, compare and evaluate the water-potability classifiers.

Methodology (all of it uses scikit-learn Pipelines, so imputation and scaling are
re-fitted on every training fold and never see validation or test rows):

* One stratified 80/20 train/test split. The test partition is only read in the final
  evaluation step, after model selection and threshold selection are finished.
* Nested stratified cross-validation on the training partition. Inner folds tune the
  hyper-parameters of Random Forest, SVC and Extra Trees (scoring: Macro F1) and the
  decision threshold; outer folds estimate performance. Every model sees identical
  outer folds and identical preprocessing.
* XGBoost's ``scale_pos_weight`` is computed inside ``fit`` from the labels it is given,
  i.e. from the training fold only.
* The decision threshold for "Potable" is chosen from out-of-fold *training* predictions
  (never from test labels) to limit false Potable predictions.
* No resampling (e.g. SMOTE) and no engineered features are used, so nothing in the
  feature matrix depends on the target.

Predictions are an educational ML estimate. They do not certify drinking-water safety.
"""
import hashlib
import json
import math
import os
import platform
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_components import FoldBalancedXGBClassifier, PotableThresholdClassifier

warnings.filterwarnings('ignore', category=UserWarning)
# scikit-learn 1.9 deprecates SVC(probability=True) (removal planned for 1.11). It is kept because
# thresholding needs Platt-scaled probabilities; requirements.txt caps scikit-learn below 1.11.
# The environment variable also reaches joblib worker processes.
warnings.filterwarnings('ignore', message='.*`probability` parameter was deprecated.*', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = ','.join(filter(None, [
    os.environ.get('PYTHONWARNINGS', ''), 'ignore::FutureWarning:sklearn.svm._base']))

# Project dataset convention: treat zero pH as missing (in memory only).
INVALID_PH_VALUE = 0

RANDOM_STATE = 42
OUTER_SEED = 42      # outer (performance-estimation) folds
INNER_SEED = 43      # inner (tuning) folds
FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
CLASS_NAMES = {0: 'Not Potable', 1: 'Potable'}

# Candidates in a fixed order; the tuned ones come first because the ensemble reuses their
# fold-local hyper-parameters.
MODEL_ORDER = ['Random_Forest', 'SVC', 'Extra_Trees', 'Soft_Voting_RF_SVC',
               'XGBoost', 'Decision_Tree', 'Logistic_Regression', 'Baseline_Majority']
SAVED_MODELS = [name for name in MODEL_ORDER if name != 'Baseline_Majority']
BASELINE = 'Baseline_Majority'
DISPLAY_NAMES = {
    'Random_Forest': 'Random Forest', 'SVC': 'SVC (RBF)', 'Extra_Trees': 'Extra Trees',
    'Soft_Voting_RF_SVC': 'Soft voting (RF + SVC)', 'XGBoost': 'XGBoost',
    'Decision_Tree': 'Decision Tree', 'Logistic_Regression': 'Logistic Regression',
    'Baseline_Majority': 'Majority baseline',
}

# Project policy parameters. They are choices, not scientific or regulatory limits.
TARGET_POTABLE_PRECISION = 0.70
MIN_PREDICTED_POTABLE = 50
THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 ... 0.90

DEFAULT_CONFIG = {
    'outer_splits': 5, 'inner_splits': 5,
    'rf_trees': 200, 'et_trees': 200, 'xgb_estimators': 200,
    'grids': {
        'Random_Forest': {'max_depth': [5, 10, 15], 'min_samples_leaf': [1, 3, 5]},
        'SVC': {'C': [0.5, 1, 2, 4], 'gamma': [0.05, 0.1, 0.2]},
        'Extra_Trees': {'min_samples_leaf': [5, 10, 20]},
    },
    'target_precision': TARGET_POTABLE_PRECISION,
    'min_predicted_potable': MIN_PREDICTED_POTABLE,
}
# Small configuration for unit tests; never used for the reported results.
FAST_CONFIG = {
    'outer_splits': 3, 'inner_splits': 3,
    'rf_trees': 15, 'et_trees': 15, 'xgb_estimators': 15,
    'grids': {
        'Random_Forest': {'max_depth': [3, 5]},
        'SVC': {'C': [1]},
        'Extra_Trees': {'min_samples_leaf': [5]},
    },
    'target_precision': TARGET_POTABLE_PRECISION,
    'min_predicted_potable': 5,
}


def log(message=''):
    print(message, flush=True)


def treat_zero_ph_as_missing(features_df):
    """Return a copy with pH == 0 converted to NaN.

    Other columns are unchanged. Callers should pass this frame to a fitted
    Pipeline; do not apply the imputer or scaler separately.
    """
    prepared = features_df.copy()
    if 'ph' not in prepared.columns:
        return prepared
    ph_numeric = pd.to_numeric(prepared['ph'], errors='coerce')
    prepared['ph'] = ph_numeric.mask(ph_numeric == INVALID_PH_VALUE, np.nan)
    return prepared


# --------------------------------------------------------------------------- data

def load_and_validate_data(file_path):
    """Load the dataset and validate its schema. Returns (dataframe, feature names)."""
    log("\n" + "=" * 80)
    log("STEP 1: LOADING AND VALIDATING DATASET")
    log("=" * 80)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    data = pd.read_csv(file_path)
    required_target = 'Potability'
    missing_cols = [col for col in FEATURES + [required_target] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    log(f"[OK] Dataset loaded: {file_path}")
    log(f"  Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    log(f"  Features (9): {', '.join(FEATURES)}")
    log(f"  Target: {required_target}")
    return data, list(FEATURES)


def report_data_quality(data, feature_cols):
    """Report data quality metrics without modifying data."""
    log("\n" + "=" * 80)
    log("STEP 2: DATA QUALITY REPORT")
    log("=" * 80)

    log("\nMissing Values:")
    missing = data[feature_cols + ['Potability']].isna().sum()
    total_cells = len(data) * len(feature_cols)
    total_missing = missing[feature_cols].sum()
    for col in feature_cols + ['Potability']:
        count = missing[col]
        status = "[WARN]" if count > 0 else "[OK]"
        log(f"  {status} {col:20s}: {count:4d} ({count / len(data) * 100:5.2f}%)")
    log(f"\n  Total missing in features: {total_missing}/{total_cells} ({total_missing / total_cells * 100:.2f}%)")
    log(f"\nDuplicate Rows: {data.duplicated().sum()}")

    counts = data['Potability'].value_counts().sort_index()
    total = len(data)
    log("\nTarget Distribution (Potability):")
    log(f"  Class 0 (Not Potable): {counts[0]:4d} ({counts[0] / total * 100:5.2f}%)")
    log(f"  Class 1 (Potable):     {counts[1]:4d} ({counts[1] / total * 100:5.2f}%)")
    imbalance_ratio = counts[0] / counts[1]
    log(f"  Imbalance ratio: {imbalance_ratio:.2f}:1 (informational; XGBoost computes its own "
        "ratio from each training fold)")

    ph_zero_count = (data['ph'] == 0).sum()
    if ph_zero_count > 0:
        log("\n[WARN] pH = 0 Detection:")
        log(f"  Found {ph_zero_count} record(s) with pH = 0")
        log("  Reason: This project treats pH = 0 as a missing-data marker")
        log("  Action: Will treat as missing value during preprocessing")
    return imbalance_ratio


def prepare_data(data, feature_cols):
    """Handle pH = 0 in memory, then make the stratified 80/20 split of raw features."""
    log("\n" + "=" * 80)
    log("STEP 3: DATA PREPARATION")
    log("=" * 80)

    ph_zero_count = int((data['ph'] == INVALID_PH_VALUE).sum())
    data_work = treat_zero_ph_as_missing(data)
    if ph_zero_count > 0:
        log("\nHandling pH = 0 (in memory, CSV unchanged):")
        log(f"  Converting {ph_zero_count} pH = 0 value(s) to NaN")

    X = data_work[feature_cols].copy()
    y = data_work['Potability'].copy()
    log(f"\n[OK] Target labels verified: {sorted(y.unique())}  (0 = Not Potable, 1 = Potable)")

    log("\nSplitting data (80/20 stratified, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    log(f"  Training set: {len(X_train):4d} samples (class 0: {(y_train == 0).sum()}, class 1: {(y_train == 1).sum()})")
    log(f"  Test set:     {len(X_test):4d} samples (class 0: {(y_test == 0).sum()}, class 1: {(y_test == 1).sum()})")
    log("  The test set is not used until the final evaluation step.")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------- pipelines

def make_classifier(name, params, config, n_jobs=1, probability=True):
    """Build an unfitted classifier. ``params`` holds the tuned hyper-parameters."""
    if name == 'Random_Forest':
        return RandomForestClassifier(n_estimators=config['rf_trees'], class_weight='balanced',
                                      random_state=RANDOM_STATE, n_jobs=n_jobs, **params)
    if name == 'SVC':
        return SVC(kernel='rbf', class_weight='balanced', probability=probability,
                   random_state=RANDOM_STATE, **params)
    if name == 'Extra_Trees':
        return ExtraTreesClassifier(n_estimators=config['et_trees'], class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=n_jobs, **params)
    if name == 'Soft_Voting_RF_SVC':
        return VotingClassifier(
            [('rf', make_classifier('Random_Forest', params['Random_Forest'], config, n_jobs)),
             ('svc', make_classifier('SVC', params['SVC'], config, probability=True))],
            voting='soft')
    if name == 'XGBoost':
        return FoldBalancedXGBClassifier(n_estimators=config['xgb_estimators'], max_depth=5,
                                         random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=n_jobs)
    if name == 'Decision_Tree':
        return DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=RANDOM_STATE)
    if name == 'Logistic_Regression':
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
    if name == BASELINE:
        return DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
    raise ValueError(f'Unknown model: {name}')


def make_pipeline(name, params, config, n_jobs=1, probability=True, deploy=False, threshold=None):
    """imputer -> scaler -> classifier. ``deploy=True`` adds the decision-threshold wrapper.

    Every candidate shares the same preprocessing. Scaling matters for SVC and Logistic
    Regression; it is harmless for tree models.
    """
    classifier = make_classifier(name, params, config, n_jobs, probability)
    if deploy:
        classifier = PotableThresholdClassifier(classifier, threshold)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', classifier),
    ])


def tune_hyperparameters(name, X, y, config, chosen):
    """Inner-CV grid search on Macro F1 using only ``X``/``y``. Returns the best parameters.

    ``chosen`` maps already-tuned model names to their parameters (used by the ensemble).
    """
    if name == 'Soft_Voting_RF_SVC':
        return {'Random_Forest': dict(chosen['Random_Forest']), 'SVC': dict(chosen['SVC'])}
    grid = config['grids'].get(name)
    if not grid:
        return {}
    search = GridSearchCV(
        make_pipeline(name, {}, config, n_jobs=1, probability=False),
        {f'classifier__{key}': values for key, values in grid.items()},
        scoring='f1_macro',
        cv=StratifiedKFold(config['inner_splits'], shuffle=True, random_state=INNER_SEED),
        n_jobs=-1, refit=False)
    search.fit(X, y)
    return {key.split('__', 1)[1]: value for key, value in search.best_params_.items()}


# ------------------------------------------------------------------------- metrics

def score_predictions(y_true, y_pred):
    """Accuracy, Macro/weighted F1 and per-class precision, recall and F1."""
    kwargs = {'labels': [0, 1], 'average': None, 'zero_division': 0}
    precision = precision_score(y_true, y_pred, **kwargs)
    recall = recall_score(y_true, y_pred, **kwargs)
    f1 = f1_score(y_true, y_pred, **kwargs)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', labels=[0, 1], zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', labels=[0, 1], zero_division=0)),
        'precision_0': float(precision[0]), 'precision_1': float(precision[1]),
        'recall_0': float(recall[0]), 'recall_1': float(recall[1]),
        'f1_0': float(f1[0]), 'f1_1': float(f1[1]),
    }


def wilson_interval(successes, n, z=1.959964):
    """95% Wilson score interval for a proportion; [None, None] when n == 0."""
    if n == 0:
        return [None, None]
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def corrected_paired_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on paired per-fold scores.

    Fold scores share training data, so an ordinary paired t-test is too optimistic.
    With only a handful of folds this test has little power: a large p-value means
    "no evidence of a difference", not "the models are equal".
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    mean = float(diff.mean())
    variance = float(diff.var(ddof=1)) if k > 1 else 0.0
    if variance == 0.0:
        return {'mean_difference': mean, 'std_difference': 0.0, 't': None,
                'p_value': 1.0 if mean == 0 else 0.0, 'folds_a_better': int((diff > 0).sum()), 'folds': k}
    t_value = mean / math.sqrt((1.0 / k + n_test / n_train) * variance)
    return {'mean_difference': mean, 'std_difference': math.sqrt(variance), 't': float(t_value),
            'p_value': float(2 * stats.t.sf(abs(t_value), k - 1)),
            'folds_a_better': int((diff > 0).sum()), 'folds': k}


# ----------------------------------------------------------------------- threshold

def threshold_curve(y_true, proba, grid=THRESHOLD_GRID):
    """Potable precision/recall and error counts at each candidate threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for threshold in grid:
        predicted = proba >= threshold
        n_predicted = int(predicted.sum())
        true_potable = int((predicted & (y_true == 1)).sum())
        rows.append({
            'threshold': float(threshold),
            'n_predicted_potable': n_predicted,
            'precision_potable': true_potable / n_predicted if n_predicted else 0.0,
            'recall_potable': true_potable / int((y_true == 1).sum()),
            'false_potable': n_predicted - true_potable,
            'false_not_potable': int((y_true == 1).sum()) - true_potable,
        })
    return rows


def select_threshold(y_true, proba, target=TARGET_POTABLE_PRECISION,
                     min_predicted=MIN_PREDICTED_POTABLE, grid=THRESHOLD_GRID):
    """Lowest threshold whose out-of-fold Potable precision reaches ``target``.

    ``y_true``/``proba`` must be out-of-fold predictions on *training* data. The lowest
    qualifying threshold keeps Potable recall as high as the precision target allows.
    A threshold must also produce at least ``min_predicted`` Potable predictions so the
    precision estimate is not based on a handful of samples. If the target cannot be met
    the model keeps its native decision rule (``threshold=None``).
    """
    curve = threshold_curve(y_true, proba, grid)
    for row in curve:
        if row['n_predicted_potable'] >= min_predicted and row['precision_potable'] >= target:
            return {'threshold': row['threshold'], 'target_met': True,
                    'oof_precision_potable': row['precision_potable'],
                    'oof_recall_potable': row['recall_potable'],
                    'oof_n_predicted_potable': row['n_predicted_potable'], 'curve': curve}
    return {'threshold': None, 'target_met': False, 'oof_precision_potable': None,
            'oof_recall_potable': None, 'oof_n_predicted_potable': None, 'curve': curve}


def apply_threshold(potable_proba, threshold, native_prediction):
    if threshold is None:
        return np.asarray(native_prediction)
    return (np.asarray(potable_proba) >= threshold).astype(int)


# ---------------------------------------------------------------- nested validation

def evaluate_fold(name, params, X_train, y_train, X_val, y_val, config):
    """Fit on one outer training fold; score on the outer validation fold.

    The decision threshold is chosen from inner out-of-fold predictions on the outer
    *training* fold only, then applied once to the outer validation fold.
    """
    pipeline = make_pipeline(name, params, config).fit(X_train, y_train)
    native = pipeline.predict(X_val)
    result = {f'val_{key}': value for key, value in score_predictions(y_val, native).items()}
    result['train_accuracy'] = float(accuracy_score(y_train, pipeline.predict(X_train)))
    if name != BASELINE:
        inner = StratifiedKFold(config['inner_splits'], shuffle=True, random_state=INNER_SEED)
        oof = cross_val_predict(make_pipeline(name, params, config), X_train, y_train, cv=inner,
                                method='predict_proba', n_jobs=-1)[:, 1]
        choice = select_threshold(y_train, oof, config['target_precision'], config['min_predicted_potable'])
        predicted = apply_threshold(pipeline.predict_proba(X_val)[:, 1], choice['threshold'], native)
        for key, value in score_predictions(y_val, predicted).items():
            result[f'thr_{key}'] = value
        result['thr_threshold'] = choice['threshold']
        result['thr_target_met'] = choice['target_met']
        result['thr_n_predicted_potable'] = int((predicted == 1).sum())
        result['thr_false_potable'] = int(((predicted == 1) & (np.asarray(y_val) == 0)).sum())
    return result


CV_METRIC_KEYS = {
    'accuracy': 'cv_accuracy', 'macro_f1': 'cv_macro_f1', 'weighted_f1': 'cv_weighted_f1',
    'precision_0': 'cv_precision_class_0', 'precision_1': 'cv_precision_class_1',
    'recall_0': 'cv_recall_class_0', 'recall_1': 'cv_recall_class_1',
    'f1_0': 'cv_f1_class_0', 'f1_1': 'cv_f1_class_1',
}


def aggregate_folds(folds):
    """Mean and population standard deviation (ddof=0, as before) across outer folds."""
    out = {}
    for metric, prefix in CV_METRIC_KEYS.items():
        values = np.array([fold[f'val_{metric}'] for fold in folds])
        out[f'{prefix}_mean'] = float(values.mean())
        out[f'{prefix}_std'] = float(values.std())
    train = np.array([fold['train_accuracy'] for fold in folds])
    out['train_accuracy_mean'] = float(train.mean())
    out['train_accuracy_std'] = float(train.std())
    out['cv_macro_f1_folds'] = [fold['val_macro_f1'] for fold in folds]
    out['cv_accuracy_folds'] = [fold['val_accuracy'] for fold in folds]
    if 'thr_accuracy' in folds[0]:
        for metric in ['accuracy', 'macro_f1', 'precision_1', 'recall_1', 'recall_0',
                       'false_potable', 'n_predicted_potable']:
            values = np.array([fold[f'thr_{metric}'] for fold in folds], dtype=float)
            out[f'cv_threshold_{metric}_mean'] = float(values.mean())
            out[f'cv_threshold_{metric}_std'] = float(values.std())
        out['cv_threshold_values'] = [fold['thr_threshold'] for fold in folds]
        out['cv_threshold_target_met_folds'] = int(sum(bool(fold['thr_target_met']) for fold in folds))
    out['cv_method'] = 'nested stratified CV (outer folds), native decision rule'
    return out


def nested_cross_validation(X_train, y_train, config):
    """Outer stratified folds estimate performance; inner folds do all tuning."""
    outer = StratifiedKFold(config['outer_splits'], shuffle=True, random_state=OUTER_SEED)
    folds = {name: [] for name in MODEL_ORDER}
    tuned = {name: [] for name in MODEL_ORDER}
    log(f"Outer folds: {config['outer_splits']}, inner folds: {config['inner_splits']} "
        "(identical outer folds and preprocessing for every model)")
    for fold, (train_index, val_index) in enumerate(outer.split(X_train, y_train), start=1):
        X_fold, y_fold = X_train.iloc[train_index], y_train.iloc[train_index]
        X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]
        chosen = {}
        log(f"\n  Outer fold {fold}/{config['outer_splits']} "
            f"(train {len(X_fold)}, validation {len(X_val)})")
        for name in MODEL_ORDER:
            chosen[name] = tune_hyperparameters(name, X_fold, y_fold, config, chosen)
            tuned[name].append(chosen[name])
            result = evaluate_fold(name, chosen[name], X_fold, y_fold, X_val, y_val, config)
            folds[name].append(result)
            log(f"    {name:20s} Macro F1 {result['val_macro_f1']:.4f}  accuracy {result['val_accuracy']:.4f}"
                + (f"  params {chosen[name]}" if chosen[name] and name != 'Soft_Voting_RF_SVC' else ''))
    return {'folds': folds, 'tuned_params': tuned,
            'results': {name: aggregate_folds(folds[name]) for name in MODEL_ORDER}}


# ----------------------------------------------------------------- final fit / test

def fit_final_pipelines(X_train, y_train, config):
    """Tune on the full training partition, choose thresholds from OOF predictions, refit."""
    chosen, info, pipelines = {}, {}, {}
    outer = StratifiedKFold(config['outer_splits'], shuffle=True, random_state=OUTER_SEED)
    for name in MODEL_ORDER:
        chosen[name] = tune_hyperparameters(name, X_train, y_train, config, chosen)
        if name == BASELINE:
            choice = {'threshold': None, 'target_met': False, 'oof_precision_potable': None,
                      'oof_recall_potable': None, 'oof_n_predicted_potable': None, 'curve': []}
        else:
            oof = cross_val_predict(make_pipeline(name, chosen[name], config), X_train, y_train,
                                    cv=outer, method='predict_proba', n_jobs=-1)[:, 1]
            choice = select_threshold(y_train, oof, config['target_precision'],
                                      config['min_predicted_potable'])
        pipelines[name] = make_pipeline(name, chosen[name], config, n_jobs=-1, deploy=True,
                                        threshold=choice['threshold']).fit(X_train, y_train)
        rule = ('native decision rule of the model' if choice['threshold'] is None
                else f"predict Potable only if P(Potable) >= {choice['threshold']:.2f}")
        info[name] = {
            'hyperparameters': chosen[name],
            'decision_threshold': choice['threshold'],
            'decision_rule': rule,
            'threshold_target_met': choice['target_met'],
            'oof_threshold_precision_potable': choice['oof_precision_potable'],
            'oof_threshold_recall_potable': choice['oof_recall_potable'],
            'oof_threshold_n_predicted_potable': choice['oof_n_predicted_potable'],
            'oof_threshold_curve': choice['curve'],
        }
        log(f"  {name:20s} {rule}; params {chosen[name] if name != 'Soft_Voting_RF_SVC' else 'RF + SVC (tuned)'}")
    return pipelines, info


def evaluate_on_test_set(pipeline, X_test, y_test):
    """Final, one-time evaluation on the untouched test partition.

    Reports the deployed decision rule (what the app uses) and, for comparison, the
    model's native rule from the same fitted estimator.
    """
    y_test = np.asarray(y_test)
    deployed = pipeline.predict(X_test)
    native = pipeline[-1].estimator_.predict(pipeline[:-1].transform(X_test))
    scores = score_predictions(y_test, deployed)
    native_scores = score_predictions(y_test, native)
    cm = confusion_matrix(y_test, deployed, labels=[0, 1])
    native_cm = confusion_matrix(y_test, native, labels=[0, 1])
    n_predicted_potable = int(cm[:, 1].sum())
    return {
        'test_accuracy': scores['accuracy'],
        'test_macro_f1': scores['macro_f1'],
        'test_weighted_f1': scores['weighted_f1'],
        'test_precision_class_0': scores['precision_0'], 'test_precision_class_1': scores['precision_1'],
        'test_recall_class_0': scores['recall_0'], 'test_recall_class_1': scores['recall_1'],
        'test_f1_class_0': scores['f1_0'], 'test_f1_class_1': scores['f1_1'],
        'test_accuracy_ci95': wilson_interval(int((deployed == y_test).sum()), len(y_test)),
        'test_precision_class_1_ci95': wilson_interval(int(cm[1, 1]), n_predicted_potable),
        'test_n_predicted_potable': n_predicted_potable,
        'test_false_potable_rate': float(cm[0, 1] / cm[0].sum()),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': {'rows': 'actual', 'columns': 'predicted',
                                    'class_0': 'Not Potable', 'class_1': 'Potable'},
        'actual_not_potable_pred_not_potable': int(cm[0, 0]),
        'actual_not_potable_pred_potable': int(cm[0, 1]),
        'actual_potable_pred_not_potable': int(cm[1, 0]),
        'actual_potable_pred_potable': int(cm[1, 1]),
        'test_classification_report': classification_report(
            y_test, deployed, labels=[0, 1], target_names=['0 = Not Potable', '1 = Potable'],
            digits=4, zero_division=0),
        'test_native_accuracy': native_scores['accuracy'],
        'test_native_macro_f1': native_scores['macro_f1'],
        'test_native_weighted_f1': native_scores['weighted_f1'],
        'test_native_precision_class_0': native_scores['precision_0'],
        'test_native_precision_class_1': native_scores['precision_1'],
        'test_native_recall_class_0': native_scores['recall_0'],
        'test_native_recall_class_1': native_scores['recall_1'],
        'test_native_confusion_matrix': native_cm.tolist(),
    }


# ---------------------------------------------------------------------- selection

def select_best_model(results, nested_folds, n_train, n_test):
    """Pick the saved candidate with the highest nested-CV Macro F1; test scores are unused."""
    ranked = sorted(SAVED_MODELS, key=lambda n: results[n]['cv_macro_f1_mean'], reverse=True)
    best = ranked[0]
    comparisons = []
    for other in ranked[1:]:
        test = corrected_paired_ttest([f['val_macro_f1'] for f in nested_folds[best]],
                                      [f['val_macro_f1'] for f in nested_folds[other]], n_train, n_test)
        comparisons.append({'model': other, **test})
    runner_up = comparisons[0]
    distinguishable = runner_up['p_value'] < 0.05
    lines = [
        f"{DISPLAY_NAMES[best]} has the highest nested-CV Macro F1 "
        f"({results[best]['cv_macro_f1_mean']:.4f} +/- {results[best]['cv_macro_f1_std']:.4f}) "
        f"among the {len(SAVED_MODELS)} saved candidates; it was chosen on validation folds only.",
        f"Margin over the runner-up ({DISPLAY_NAMES[runner_up['model']]}): "
        f"{runner_up['mean_difference']:+.4f} Macro F1 (fold-to-fold SD of the difference "
        f"{runner_up['std_difference']:.4f}; corrected paired t-test p = {runner_up['p_value']:.3f}).",
    ]
    if distinguishable:
        lines.append("The corrected test flags this margin as unlikely under chance alone (p < 0.05), "
                     "but only 5 folds and 6 comparisons were available, so treat it cautiously.")
    else:
        lines.append("This margin is NOT distinguishable from chance: the corrected paired test gives "
                     "p >= 0.05. The selected model is the best point estimate, not a demonstrably "
                     "better model.")
    indistinguishable = [c['model'] for c in comparisons if c['p_value'] >= 0.05]
    if indistinguishable:
        lines.append("Candidates not statistically distinguishable from the selected model: "
                     + ', '.join(DISPLAY_NAMES[m] for m in indistinguishable) + '.')
    return {
        'selected_model': best, 'selection_metric': 'nested-CV Macro F1 (mean over outer folds)',
        'ranking': [{'model': n, 'cv_macro_f1_mean': results[n]['cv_macro_f1_mean'],
                     'cv_macro_f1_std': results[n]['cv_macro_f1_std']} for n in ranked],
        'paired_comparisons_vs_selected': comparisons,
        'selected_distinguishable_from_runner_up': distinguishable,
        'justification': lines,
    }


def build_before_after(results, baseline_results):
    """Side-by-side comparison with the original (pre-change) evaluation, if present."""
    if not baseline_results:
        return []
    rows = []
    for name in MODEL_ORDER:
        before = baseline_results.get(name)
        after = results[name]
        rows.append({
            'model': name, 'in_original_run': before is not None,
            'before_cv_macro_f1': before['cv_macro_f1_mean'] if before else None,
            'after_cv_macro_f1': after['cv_macro_f1_mean'],
            'before_test_accuracy': before['test_accuracy'] if before else None,
            'after_test_accuracy': after['test_accuracy'],
            'after_test_accuracy_native': after['test_native_accuracy'],
            'before_test_macro_f1': before['test_macro_f1'] if before else None,
            'after_test_macro_f1': after['test_macro_f1'],
            'after_test_macro_f1_native': after['test_native_macro_f1'],
            'before_test_precision_potable': before['test_precision_class_1'] if before else None,
            'after_test_precision_potable': after['test_precision_class_1'],
            'before_test_recall_potable': before['test_recall_class_1'] if before else None,
            'after_test_recall_potable': after['test_recall_class_1'],
            'before_false_potable': before['actual_not_potable_pred_potable'] if before else None,
            'after_false_potable': after['actual_not_potable_pred_potable'],
        })
    return rows


def describe_change(selected, results, baseline_results):
    """Plain statement of whether performance improved relative to the original run."""
    if not baseline_results or 'Random_Forest' not in baseline_results:
        return 'No original evaluation was available for comparison.'
    old = baseline_results['Random_Forest']
    new = results[selected]
    cv_delta = new['cv_macro_f1_mean'] - old['cv_macro_f1_mean']
    native_delta = new['test_native_macro_f1'] - old['test_macro_f1']
    deployed_delta = new['test_macro_f1'] - old['test_macro_f1']
    acc_delta = new['test_accuracy'] - old['test_accuracy']
    fp_old = old['actual_not_potable_pred_potable']
    fp_new = new['actual_not_potable_pred_potable']
    verdict = lambda d, noise: ('improved' if d > noise else 'worsened' if d < -noise else 'did not change meaningfully')
    return (
        f"Versus the original Random Forest ({old['cv_macro_f1_mean']:.4f} CV Macro F1, "
        f"{old['test_macro_f1']:.4f} test Macro F1, {old['test_accuracy']:.4f} test accuracy), the selected "
        f"{DISPLAY_NAMES[selected]}: CV Macro F1 {cv_delta:+.4f} ({verdict(cv_delta, new['cv_macro_f1_std'])} "
        f"relative to its fold SD {new['cv_macro_f1_std']:.4f}); test Macro F1 {native_delta:+.4f} under its native "
        f"rule and {deployed_delta:+.4f} under the deployed safety threshold; test accuracy {acc_delta:+.4f} "
        f"(deployed). Test records labelled Not Potable predicted Potable: {fp_old} originally, {fp_new} now. "
        "Differences of about one fold standard deviation or less (and a single 656-record test split) are "
        "not evidence of real improvement.")


# ---------------------------------------------------------------------- experiment

def run_experiment(X_train, X_test, y_train, y_test, config, baseline_results=None):
    """Select with training data only; touch the test partition only for final scoring."""
    log("\n" + "=" * 80)
    log("STEP 4: NESTED STRATIFIED CROSS-VALIDATION (model selection)")
    log("=" * 80)
    nested = nested_cross_validation(X_train, y_train, config)
    results = {name: dict(values) for name, values in nested['results'].items()}

    log("\n" + "=" * 80)
    log("STEP 5: SELECTION, FINAL FIT AND DECISION THRESHOLDS (training data only)")
    log("=" * 80)
    n_test_fold = len(X_train) // config['outer_splits']
    selection = select_best_model(results, nested['folds'], len(X_train) - n_test_fold, n_test_fold)
    for line in selection['justification']:
        log(f"  {line}")
    pipelines, final_info = fit_final_pipelines(X_train, y_train, config)

    log("\n" + "=" * 80)
    log("STEP 6: FINAL EVALUATION ON THE UNTOUCHED TEST SET")
    log("=" * 80)
    for name in MODEL_ORDER:
        results[name].update(final_info[name])
        results[name].update(evaluate_on_test_set(pipelines[name], X_test, y_test))
        results[name]['model_name'] = name
        log(f"  {name:20s} accuracy {results[name]['test_accuracy']:.4f}  Macro F1 {results[name]['test_macro_f1']:.4f}  "
            f"(native rule: {results[name]['test_native_accuracy']:.4f} / {results[name]['test_native_macro_f1']:.4f})  "
            f"false Potable {results[name]['actual_not_potable_pred_potable']}")

    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'config': config,
        'split': {
            'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
            'train_class_counts': {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            'test_class_counts': {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        'selection': selection,
        'nested_fold_scores': {name: {'macro_f1': results[name]['cv_macro_f1_folds'],
                                      'accuracy': results[name]['cv_accuracy_folds']} for name in MODEL_ORDER},
        'nested_selected_params': {name: nested['tuned_params'][name] for name in MODEL_ORDER},
        'threshold_policy': {
            'target_potable_precision': config['target_precision'],
            'min_predicted_potable': config['min_predicted_potable'],
            'grid': THRESHOLD_GRID,
            'rule': 'lowest threshold on out-of-fold TRAINING predictions whose Potable precision >= target',
        },
        'before_after': build_before_after(results, baseline_results),
        'change_statement': describe_change(selection['selected_model'], results, baseline_results),
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    return pipelines, results, summary


# ------------------------------------------------------------------------- output

def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def pm(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def fmt(value, digits=4):
    return 'n/a' if value is None else f"{value:.{digits}f}"


def confusion_text(cm, indent='  '):
    return (f"{indent}Label 0 = Not Potable, Label 1 = Potable\n"
            f"{indent}                     Pred Not Potable (0)  Pred Potable (1)\n"
            f"{indent}Actual Not Potable (0)       {cm[0][0]:6d}              {cm[0][1]:6d}\n"
            f"{indent}Actual Potable (1)           {cm[1][0]:6d}              {cm[1][1]:6d}\n")


def format_report(results, summary):
    selected = summary['selection']['selected_model']
    policy = summary['threshold_policy']
    split = summary['split']
    out = []
    w = out.append
    line = "=" * 80
    w(line)
    w("WATER QUALITY PREDICTION - MODEL EVALUATION REPORT")
    w(f"Generated: {summary['generated'].replace('T', ' ')}")
    w(line)
    w("")
    w("TRAINING CONFIGURATION:")
    w("  - Dataset: dataset/water_potability.csv (read-only; never rewritten)")
    w(f"  - Features: 9 ({', '.join(FEATURES)})")
    w("  - Target: Potability (0=Not Potable, 1=Potable)")
    w(f"  - Train/Test Split: 80/20 stratified, random_state=42 "
      f"({split['train_size']} train / {split['test_size']} test)")
    w("  - Preprocessing Pipeline: median SimpleImputer -> StandardScaler -> classifier, fitted")
    w("    independently inside every training fold (never on validation or test rows)")
    w("  - pH=0 Handling: converted to missing in memory (CSV unchanged)")
    w(f"  - Model selection: nested stratified CV ({summary['config']['outer_splits']} outer x "
      f"{summary['config']['inner_splits']} inner folds); Macro F1 is the primary metric")
    w("  - Tuned (inner CV, Macro F1): Random Forest, SVC (RBF), Extra Trees; soft voting reuses the")
    w("    tuned RF and SVC parameters. XGBoost, Decision Tree and Logistic Regression are untuned.")
    w("  - XGBoost scale_pos_weight: computed from each training fold's labels only (no leakage)")
    w("  - No resampling (SMOTE etc.) and no engineered features are used")
    w("  - Test set: untouched until the final evaluation; thresholds never use test labels")
    w("")
    w(f"SELECTED MODEL: {selected}  ({DISPLAY_NAMES[selected]})")
    w(f"  Selection metric: {summary['selection']['selection_metric']}")
    for text in summary['selection']['justification']:
        w(f"  - {text}")
    w("")
    w("  Ranking by nested-CV Macro F1:")
    for position, row in enumerate(summary['selection']['ranking'], start=1):
        w(f"    {position}. {row['model']:22s} {pm(row['cv_macro_f1_mean'], row['cv_macro_f1_std'])}")
    w("")
    w("  Paired comparison of the selected model against each other candidate")
    w("  (same outer folds; Nadeau-Bengio corrected resampled t-test on Macro F1):")
    w("    Candidate               mean diff   SD diff   p-value   folds selected is better")
    for c in summary['selection']['paired_comparisons_vs_selected']:
        w(f"    {c['model']:22s} {c['mean_difference']:+9.4f} {c['std_difference']:9.4f} "
          f"{c['p_value']:9.3f}   {c['folds_a_better']}/{c['folds']}")
    w("    With 5 folds this test has little power. p >= 0.05 means 'no evidence of a difference',")
    w("    not 'equal'. Six comparisons are uncorrected for multiplicity.")
    w("")
    w(line)
    w("NESTED CROSS-VALIDATION RESULTS (mean +/- SD over outer folds, native decision rule)")
    w(line)
    w("  Model                   Macro F1          Accuracy          P(Potable)        R(Potable)")
    for name in MODEL_ORDER:
        r = results[name]
        w(f"  {name:22s} {pm(r['cv_macro_f1_mean'], r['cv_macro_f1_std'])}  "
          f"{pm(r['cv_accuracy_mean'], r['cv_accuracy_std'])}  "
          f"{pm(r['cv_precision_class_1_mean'], r['cv_precision_class_1_std'])}  "
          f"{pm(r['cv_recall_class_1_mean'], r['cv_recall_class_1_std'])}")
    w("  SDs are population SDs over folds (ddof=0), not confidence intervals.")
    w("")
    w("  Per-class F1 and train-versus-validation gap (accuracy):")
    w("  Model                   F1 Not Potable    F1 Potable        Train acc         Gap")
    for name in MODEL_ORDER:
        r = results[name]
        w(f"  {name:22s} {pm(r['cv_f1_class_0_mean'], r['cv_f1_class_0_std'])}  "
          f"{pm(r['cv_f1_class_1_mean'], r['cv_f1_class_1_std'])}  "
          f"{pm(r['train_accuracy_mean'], r['train_accuracy_std'])}  "
          f"{r['train_accuracy_mean'] - r['cv_accuracy_mean']:.4f}")
    w("  A large gap means the model fits its training folds far better than unseen folds (overfitting).")
    w("")
    w(line)
    w("DECISION THRESHOLD FOR 'POTABLE' (prioritising fewer false Potable predictions)")
    w(line)
    w("  Rationale: a false Potable prediction (a sample labelled Not Potable that the model calls")
    w("  Potable) is the costlier error for a water-quality screen, so the model is asked to call")
    w("  a sample Potable only when its Potable probability is high enough. The threshold is the")
    w(f"  LOWEST value whose out-of-fold Potable precision on TRAINING data is >= {policy['target_potable_precision']:.2f}")
    w(f"  (with at least {policy['min_predicted_potable']} Potable predictions behind the estimate),")
    w("  which keeps Potable recall as high as that precision target permits. The 0.70 target is a")
    w("  project policy choice, not a scientific or regulatory limit. If a model cannot reach it,")
    w("  it keeps its native decision rule. No test label is used to choose any threshold.")
    w("")
    w("  Trade-off: raising the threshold lowers false Potable predictions but raises false Not")
    w("  Potable predictions (Potable samples that are rejected), so Potable recall, Macro F1 and")
    w("  often accuracy fall. Both errors are reported; neither is free.")
    w("")
    w("  Threshold chosen from out-of-fold training predictions (final models):")
    w("  Model                   threshold  OOF P(Potable)  OOF R(Potable)  OOF #Potable  target met")
    for name in SAVED_MODELS:
        r = results[name]
        threshold = 'native' if r['decision_threshold'] is None else f"{r['decision_threshold']:.2f}"
        w(f"  {name:22s} {threshold:>9s}  {fmt(r['oof_threshold_precision_potable']):>14s}  "
          f"{fmt(r['oof_threshold_recall_potable']):>14s}  "
          f"{str(r['oof_threshold_n_predicted_potable'] if r['oof_threshold_n_predicted_potable'] is not None else 'n/a'):>12s}  "
          f"{'yes' if r['threshold_target_met'] else 'no'}")
    w("")
    w("  Nested estimate of the threshold policy (threshold re-chosen inside each outer training")
    w("  fold, applied once to that fold's held-out rows):")
    w("  Model                   P(Potable)        R(Potable)        false Potable/fold  Macro F1")
    for name in SAVED_MODELS:
        r = results[name]
        w(f"  {name:22s} {pm(r['cv_threshold_precision_1_mean'], r['cv_threshold_precision_1_std'])}  "
          f"{pm(r['cv_threshold_recall_1_mean'], r['cv_threshold_recall_1_std'])}  "
          f"{r['cv_threshold_false_potable_mean']:10.1f}         {r['cv_threshold_macro_f1_mean']:.4f}")
    w("  Held-out precision can sit below the 0.70 target: the threshold was chosen on data the model")
    w("  was tuned on, and the number of Potable predictions per fold is small.")
    w("")
    w(f"  Out-of-fold curve for the selected model ({selected}), training data only:")
    w("  threshold  #pred Potable  precision(Potable)  recall(Potable)  false Potable  false Not Potable")
    for row in results[selected]['oof_threshold_curve']:
        w(f"  {row['threshold']:9.2f}  {row['n_predicted_potable']:13d}  {row['precision_potable']:18.4f}  "
          f"{row['recall_potable']:15.4f}  {row['false_potable']:13d}  {row['false_not_potable']:17d}")
    w("")

    for name in MODEL_ORDER:
        r = results[name]
        w(line)
        w(f"MODEL: {name}" + ("   <-- SELECTED" if name == selected else ""))
        w(line)
        w("")
        if r['hyperparameters']:
            w(f"Tuned hyper-parameters (full training set): {r['hyperparameters']}")
        w(f"Decision rule used by the app: {r['decision_rule']}")
        w("")
        w("Nested Cross-Validation Results (native rule):")
        w(f"  Accuracy:    {pm(r['cv_accuracy_mean'], r['cv_accuracy_std'])}")
        w(f"  Macro F1:    {pm(r['cv_macro_f1_mean'], r['cv_macro_f1_std'])}")
        w(f"  Weighted F1: {pm(r['cv_weighted_f1_mean'], r['cv_weighted_f1_std'])}")
        w(f"  Train Acc:   {pm(r['train_accuracy_mean'], r['train_accuracy_std'])}")
        w(f"  Overfit Gap: {r['train_accuracy_mean'] - r['cv_accuracy_mean']:.4f}")
        w("")
        w("Test Set Results (deployed decision rule):")
        w(f"  Accuracy:    {r['test_accuracy']:.4f}   (95% Wilson CI {fmt(r['test_accuracy_ci95'][0])} - {fmt(r['test_accuracy_ci95'][1])})")
        w(f"  Macro F1:    {r['test_macro_f1']:.4f}")
        w(f"  Weighted F1: {r['test_weighted_f1']:.4f}")
        w(f"  Potable precision 95% CI: {fmt(r['test_precision_class_1_ci95'][0])} - "
          f"{fmt(r['test_precision_class_1_ci95'][1])} ({r['test_n_predicted_potable']} Potable predictions)")
        w(f"  Share of Not Potable records predicted Potable: {r['test_false_potable_rate']:.4f}")
        w("")
        w("Classification report (test set, deployed rule):")
        for text_line in r['test_classification_report'].splitlines():
            w("  " + text_line)
        w("")
        w("Confusion Matrix (test set, deployed rule; rows=actual, cols=predicted):")
        w(confusion_text(r['confusion_matrix']).rstrip('\n'))
        w("")
        w("Same fitted model under its native decision rule (for comparison only):")
        w(f"  Accuracy {r['test_native_accuracy']:.4f}, Macro F1 {r['test_native_macro_f1']:.4f}, "
          f"Weighted F1 {r['test_native_weighted_f1']:.4f}, Potable precision {r['test_native_precision_class_1']:.4f}, "
          f"Potable recall {r['test_native_recall_class_1']:.4f}")
        w(confusion_text(r['test_native_confusion_matrix']).rstrip('\n'))
        w("")

    w(line)
    w("BEFORE / AFTER COMPARISON WITH THE ORIGINAL EVALUATION")
    w(line)
    w("  'Before' values come from models/baseline_evaluation_results.json (the original run, kept")
    w("  unchanged). 'After' test values use the deployed decision rule; 'native' uses each model's own.")
    w("  Original CV figures were plain 5-fold CV; new CV figures are nested-CV outer folds.")
    w("")
    w("  Model                   CV MacroF1 before->after   Test acc before->after(native)   Test MacroF1 before->after(native)")
    for row in summary['before_after']:
        w(f"  {row['model']:22s} {fmt(row['before_cv_macro_f1']):>7s} -> {row['after_cv_macro_f1']:.4f}"
          f"      {fmt(row['before_test_accuracy']):>7s} -> {row['after_test_accuracy']:.4f} ({row['after_test_accuracy_native']:.4f})"
          f"      {fmt(row['before_test_macro_f1']):>7s} -> {row['after_test_macro_f1']:.4f} ({row['after_test_macro_f1_native']:.4f})")
    w("")
    w("  Model                   Potable precision before->after   Potable recall before->after   false Potable before->after")
    for row in summary['before_after']:
        w(f"  {row['model']:22s} {fmt(row['before_test_precision_potable']):>7s} -> {row['after_test_precision_potable']:.4f}"
          f"            {fmt(row['before_test_recall_potable']):>7s} -> {row['after_test_recall_potable']:.4f}"
          f"            {str(row['before_false_potable'] if row['before_false_potable'] is not None else 'n/a'):>5s} -> {row['after_false_potable']}")
    w("")
    w(f"  {summary['change_statement']}")
    w("")
    w(line)
    w("LIMITATIONS AND UNCERTAINTY:")
    w(line)
    w("1. The features carry weak signal (all absolute correlations with Potability are below 0.04).")
    w("   Every model remains close to the majority baseline in accuracy; accuracy alone says little.")
    w("2. Results come from one dataset, one train/test split and one nested-CV repetition. The test")
    w("   set has 656 records, so test accuracy carries roughly +/-3.7 percentage points of sampling")
    w("   uncertainty (see the Wilson intervals above). Fold scores are correlated; SDs over 5 folds")
    w("   are rough guides, not confidence intervals.")
    w("3. Differences between candidates are small relative to that noise unless the paired test says")
    w("   otherwise. Selecting among several candidates and a hyper-parameter grid still carries some")
    w("   selection optimism despite the nested design.")
    w("4. The threshold target (Potable precision >= 0.70) is a project choice, not a standard. Final")
    w("   thresholds were chosen on out-of-fold training predictions using hyper-parameters tuned on")
    w("   the same training data, so held-out precision can be lower. Raising the threshold reduces")
    w("   false Potable predictions but rejects many samples labelled Potable.")
    w("5. Probabilities are model outputs, not calibrated real-world confidence. SVC probabilities are")
    w("   Platt-scaled; tree-ensemble probabilities are vote shares.")
    w("6. Median imputation fills 14.99% (pH), 23.84% (Sulfate) and 4.95% (Trihalomethanes) missing")
    w("   values; tree models and SVC can overfit the imputed values. Scaling is required for SVC and")
    w("   Logistic Regression and has no effect on tree splits.")
    w("7. There is no external, temporal or geographic validation. Results do not establish scientific,")
    w("   medical or production readiness.")
    w("")
    w(line)
    w("IMPORTANT NOTES:")
    w(line)
    w("1. Model predictions are for educational purposes only")
    w("2. DO NOT use as sole certification for drinking water safety")
    w("3. Always follow official water quality testing standards (EPA, WHO)")
    w("4. Large train-validation gaps indicate overfitting")
    w("5. Macro F1 is the primary selection metric for this imbalanced classification task")
    w("6. All preprocessing is embedded in saved pipelines")
    w("")
    return "\n".join(out)


def dataset_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def save_artifacts(pipelines, results, summary, feature_names, output_dir='models'):
    """Save fitted Pipelines, evaluation JSON, the readable report and the training summary."""
    log("\n" + "=" * 80)
    log("STEP 7: SAVING ARTIFACTS")
    log("=" * 80)
    os.makedirs(output_dir, exist_ok=True)

    log("\nSaving trained pipelines:")
    for name in SAVED_MODELS:
        filepath = f"{output_dir}/{name}.pkl"
        joblib.dump(pipelines[name], filepath, compress=3)
        log(f"  [OK] {name:25s} -> {filepath:40s} ({os.path.getsize(filepath) / 1024:>8.1f} KB)")

    feature_names_path = f"{output_dir}/feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    log(f"  [OK] Feature names -> {feature_names_path}")

    results_json_path = f"{output_dir}/evaluation_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as handle:
        json.dump(to_serializable(results), handle, indent=2)
    log(f"  [OK] Evaluation results -> {results_json_path}")

    summary_path = f"{output_dir}/training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(to_serializable(summary), handle, indent=2)
    log(f"  [OK] Training summary -> {summary_path}")

    report_path = f"{output_dir}/evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(format_report(results, summary))
    log(f"  [OK] Evaluation report -> {report_path}")
    log(f"\n[OK] All artifacts saved to {output_dir}/")
    return results_json_path, report_path


def load_baseline_results(path='models/baseline_evaluation_results.json'):
    """The original evaluation, kept for comparison. Never overwritten by this script."""
    if not os.path.exists(path):
        log(f"[WARN] {path} not found; the report will omit the before/after comparison.")
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def main():
    """Main training pipeline."""
    log("\n" + "=" * 80)
    log("WATER QUALITY PREDICTION - NESTED-CV PIPELINE TRAINING")
    log("=" * 80)
    log("Pipelines with fold-local preprocessing, nested CV, and training-only thresholds")
    log("=" * 80)

    data_path = 'dataset/water_potability.csv'
    data, feature_cols = load_and_validate_data(data_path)
    report_data_quality(data, feature_cols)
    X_train, X_test, y_train, y_test = prepare_data(data, feature_cols)

    baseline_results = load_baseline_results()
    pipelines, results, summary = run_experiment(
        X_train, X_test, y_train, y_test, DEFAULT_CONFIG, baseline_results)
    summary['dataset'] = {'path': data_path, 'sha256': dataset_sha256(data_path), 'rows': int(len(data))}
    results_path, report_path = save_artifacts(pipelines, results, summary, feature_cols)

    best = summary['selection']['selected_model']
    log("\n" + "=" * 80)
    log("TRAINING COMPLETE")
    log("=" * 80)
    log(f"[OK] Selected model: {best} ({summary['selection']['selection_metric']})")
    log(f"[OK] Nested-CV Macro F1: {results[best]['cv_macro_f1_mean']:.4f} +/- {results[best]['cv_macro_f1_std']:.4f}")
    log(f"[OK] Test accuracy (deployed rule): {results[best]['test_accuracy']:.4f}")
    log(f"[OK] Decision rule: {results[best]['decision_rule']}")
    log(f"[OK] {summary['change_statement']}")
    log(f"[OK] Evaluation report: {report_path}")
    log("\nNext steps:")
    log(f"  1. Review: {report_path}")
    log("  2. Test: python test_models.py && python test_training.py && python test_app.py")
    log("  3. Run app: python -m streamlit run app.py")
    log("=" * 80)


if __name__ == "__main__":
    main()
