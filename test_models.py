"""
Test script to verify Pipeline models load correctly and make predictions.
Exits with a non-zero status if any required check fails.
"""
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from train_model import treat_zero_ph_as_missing

EXPECTED_MODELS = ['Random_Forest', 'XGBoost', 'Decision_Tree', 'Logistic_Regression']
EXPECTED_STEPS = ['imputer', 'scaler', 'classifier']
EXPECTED_FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
VALID_CLASSES = {0, 1}

failures = []
checks_run = 0
checks_passed = 0


def record(condition, pass_msg, fail_msg):
    global checks_run, checks_passed
    checks_run += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {pass_msg}")
        return True
    print(f"  [FAIL] {fail_msg}")
    failures.append(fail_msg)
    return False


def assert_valid_proba(proba, model_name, context):
    proba = np.asarray(proba)
    record(
        proba.ndim == 1 and proba.shape[0] == 2,
        f"{model_name} {context}: probability shape is {proba.shape}",
        f"{model_name} {context}: expected probability shape (2,), got {proba.shape}",
    )
    record(
        np.all(np.isfinite(proba)),
        f"{model_name} {context}: probabilities are finite",
        f"{model_name} {context}: probabilities contain NaN or Inf: {proba}",
    )
    record(
        np.isclose(np.sum(proba), 1.0, atol=1e-5),
        f"{model_name} {context}: probabilities sum to {np.sum(proba):.6f}",
        f"{model_name} {context}: probabilities sum to {np.sum(proba)}, expected ~1",
    )
    record(
        np.all((proba >= 0) & (proba <= 1)),
        f"{model_name} {context}: probabilities are in [0, 1]",
        f"{model_name} {context}: probabilities out of range: {proba}",
    )


print("=" * 80)
print("PIPELINE MODEL TESTING")
print("=" * 80)

print("\n1. Testing Pipeline loading...")
pipelines = {}
for model_name in EXPECTED_MODELS:
    path = f'models/{model_name}.pkl'
    try:
        pipeline = joblib.load(path)
        pipelines[model_name] = pipeline
        record(True, f"{model_name:25s} loaded from {path}", "")
    except Exception as e:
        record(False, "", f"{model_name} failed to load from {path}: {e}")

record(
    all(name in pipelines for name in EXPECTED_MODELS),
    "All four saved model files loaded",
    f"Missing models: {[name for name in EXPECTED_MODELS if name not in pipelines]}",
)

print("\n2. Verifying fitted scikit-learn Pipelines and steps...")
for model_name in EXPECTED_MODELS:
    if model_name not in pipelines:
        record(False, "", f"{model_name} skipped pipeline checks because it did not load")
        continue

    pipeline = pipelines[model_name]
    record(
        isinstance(pipeline, Pipeline),
        f"{model_name} is a sklearn Pipeline",
        f"{model_name} is {type(pipeline).__name__}, expected sklearn.pipeline.Pipeline",
    )
    if not isinstance(pipeline, Pipeline):
        continue

    steps = list(pipeline.named_steps.keys())
    record(
        steps == EXPECTED_STEPS,
        f"{model_name} steps: {steps}",
        f"{model_name} steps {steps} do not match {EXPECTED_STEPS}",
    )

    try:
        check_is_fitted(pipeline)
        fitted = True
        fitted_error = ""
    except Exception as e:
        fitted = False
        fitted_error = str(e)
    record(
        fitted,
        f"{model_name} is fitted",
        f"{model_name} is not a fitted pipeline: {fitted_error}",
    )

print("\n3. Testing feature names and order...")
feature_names = None
try:
    feature_names = joblib.load('models/feature_names.pkl')
    record(True, f"Feature names loaded: {feature_names}", "")
except Exception as e:
    record(False, "", f"Feature names failed to load: {e}")

record(
    feature_names == EXPECTED_FEATURES,
    "Feature names and order match the project schema",
    f"Feature mismatch. Expected {EXPECTED_FEATURES}, got {feature_names}",
)

print("\n4. Testing predictions with a complete sample...")
print("   Sample: ph=7.0, Hardness=200, Solids=20000, Chloramines=7, Sulfate=330,")
print("           Conductivity=420, Organic_carbon=14, Trihalomethanes=66, Turbidity=4")

if feature_names != EXPECTED_FEATURES:
    record(False, "", "Skipped complete-sample predictions because feature names are invalid")
else:
    sample_df = pd.DataFrame(
        [[7.0, 200.0, 20000.0, 7.0, 330.0, 420.0, 14.0, 66.0, 4.0]],
        columns=feature_names,
    )
    for model_name in EXPECTED_MODELS:
        if model_name not in pipelines:
            record(False, "", f"{model_name} skipped complete-sample prediction because it did not load")
            continue
        pipeline = pipelines[model_name]
        try:
            prediction = pipeline.predict(sample_df)
            proba = pipeline.predict_proba(sample_df)
        except Exception as e:
            record(False, "", f"{model_name} complete-sample prediction raised: {e}")
            continue

        pred_value = prediction[0]
        record(
            pred_value in VALID_CLASSES,
            f"{model_name} predicted class {int(pred_value)}",
            f"{model_name} predicted invalid class {pred_value}",
        )
        record(
            np.asarray(proba).shape == (1, 2),
            f"{model_name} predict_proba shape is {np.asarray(proba).shape}",
            f"{model_name} predict_proba shape {np.asarray(proba).shape} != (1, 2)",
        )
        assert_valid_proba(np.asarray(proba)[0], model_name, "complete sample")

print("\n5. Testing missing-value handling (ph=NaN) for every model...")
if feature_names != EXPECTED_FEATURES:
    record(False, "", "Skipped missing-value predictions because feature names are invalid")
else:
    missing_df = pd.DataFrame(
        [[np.nan, 200.0, 20000.0, 7.0, 330.0, 420.0, 14.0, 66.0, 4.0]],
        columns=feature_names,
    )
    for model_name in EXPECTED_MODELS:
        if model_name not in pipelines:
            record(False, "", f"{model_name} skipped missing-value prediction because it did not load")
            continue
        pipeline = pipelines[model_name]
        try:
            prediction = pipeline.predict(missing_df)
            proba = pipeline.predict_proba(missing_df)
        except Exception as e:
            record(False, "", f"{model_name} missing-value prediction raised: {e}")
            continue

        pred_value = prediction[0]
        record(
            pred_value in VALID_CLASSES,
            f"{model_name} handled ph=NaN and predicted class {int(pred_value)}",
            f"{model_name} ph=NaN produced invalid class {pred_value}",
        )
        record(
            np.asarray(proba).shape == (1, 2),
            f"{model_name} ph=NaN predict_proba shape is {np.asarray(proba).shape}",
            f"{model_name} ph=NaN predict_proba shape {np.asarray(proba).shape} != (1, 2)",
        )
        assert_valid_proba(np.asarray(proba)[0], model_name, "ph=NaN")

print("\n6. Testing zero pH vs missing pH predictions for every model...")
if feature_names != EXPECTED_FEATURES:
    record(False, "", "Skipped zero/NaN equivalence because feature names are invalid")
else:
    zero_df = missing_df.copy(deep=True)
    zero_df['ph'] = 0.0
    zero_before = zero_df.copy(deep=True)
    missing_before = missing_df.copy(deep=True)
    prepared_zero = treat_zero_ph_as_missing(zero_df)
    prepared_missing = treat_zero_ph_as_missing(missing_df)
    for model_name in EXPECTED_MODELS:
        if model_name not in pipelines:
            record(False, "", f"{model_name} skipped zero/NaN equivalence because it did not load")
            continue
        pipeline = pipelines[model_name]
        try:
            zero_prediction = np.asarray(pipeline.predict(prepared_zero))
            missing_prediction = np.asarray(pipeline.predict(prepared_missing))
            zero_proba = np.asarray(pipeline.predict_proba(prepared_zero))
            missing_proba = np.asarray(pipeline.predict_proba(prepared_missing))
            record(
                zero_prediction.shape == missing_prediction.shape == (1,)
                and np.array_equal(zero_prediction, missing_prediction)
                and np.isin(zero_prediction, list(VALID_CLASSES)).all(),
                f"{model_name} zero pH and NaN give the same valid prediction",
                f"{model_name} zero/NaN predictions differ or are invalid: "
                f"{zero_prediction} vs {missing_prediction}",
            )
            np.testing.assert_allclose(zero_proba, missing_proba, rtol=1e-7, atol=1e-9)
            record(
                zero_proba.shape == missing_proba.shape == (1, 2),
                f"{model_name} zero pH and NaN give matching probabilities of shape (1, 2)",
                f"{model_name} zero/NaN probability shapes are invalid",
            )
            assert_valid_proba(zero_proba[0], model_name, "ph=0 via shared helper")
        except Exception as e:
            record(False, "", f"{model_name} zero/NaN equivalence failed: {e}")
    try:
        pd.testing.assert_frame_equal(zero_df, zero_before)
        pd.testing.assert_frame_equal(missing_df, missing_before)
        record(True, "Zero/NaN inference leaves both original inputs unchanged", "")
    except AssertionError as e:
        record(False, "", f"Zero/NaN inference mutated original inputs: {e}")

print("\n7. Testing mixed-batch pH handling and nonmutation...")
batch_df = pd.DataFrame(
    [
        [0.0, 100.0, 20000.0, 7.0, 330.0, 420.0, 14.0, 66.0, 4.0],
        [np.nan, 100.0, 20000.0, 7.0, 330.0, 420.0, 14.0, 66.0, 4.0],
        [7.2, 200.0, 15000.0, 6.0, np.nan, 350.0, 12.0, 50.0, 3.0],
        [0.0, 0.0, 10000.0, 8.0, 300.0, 400.0, 10.0, np.nan, 2.0],
        [np.nan, 0.0, 10000.0, 8.0, 300.0, 400.0, 10.0, np.nan, 2.0],
        [8.4, 180.0, 25000.0, 5.0, 340.0, 450.0, 16.0, 70.0, 5.0],
    ],
    columns=EXPECTED_FEATURES,
    index=pd.Index([11, 23, 35, 47, 59, 71], name='sample'),
)
batch_before = batch_df.copy(deep=True)
expected_batch = batch_df.copy(deep=True)
expected_batch.loc[[11, 47], 'ph'] = np.nan
try:
    prepared_batch = treat_zero_ph_as_missing(batch_df)
    pd.testing.assert_frame_equal(prepared_batch, expected_batch)
    record(True, "Batch helper converts only zero pH, preserving missing values, other columns and index", "")
except Exception as e:
    prepared_batch = None
    record(False, "", f"Batch pH conversion failed: {e}")

for model_name in EXPECTED_MODELS:
    if model_name not in pipelines or prepared_batch is None:
        record(False, "", f"{model_name} skipped mixed-batch prediction: model or prepared batch unavailable")
        continue
    pipeline = pipelines[model_name]
    try:
        prediction = np.asarray(pipeline.predict(prepared_batch))
        proba = np.asarray(pipeline.predict_proba(prepared_batch))
        np.testing.assert_equal(prediction.shape, (len(batch_df),))
        np.testing.assert_equal(proba.shape, (len(batch_df), 2))
        record(
            np.isin(prediction, list(VALID_CLASSES)).all(),
            f"{model_name} mixed batch produces one valid prediction per row",
            f"{model_name} mixed batch has invalid predictions: {prediction}",
        )
        for zero_row, missing_row in [(0, 1), (3, 4)]:
            np.testing.assert_equal(prediction[zero_row], prediction[missing_row])
            np.testing.assert_allclose(proba[zero_row], proba[missing_row], rtol=1e-7, atol=1e-9)
        for row in range(len(batch_df)):
            single = treat_zero_ph_as_missing(batch_df.iloc[[row]])
            np.testing.assert_array_equal(pipeline.predict(single), prediction[row:row + 1])
            np.testing.assert_allclose(
                pipeline.predict_proba(single), proba[row:row + 1], rtol=1e-7, atol=1e-9,
            )
            assert_valid_proba(proba[row], model_name, f"batch row {row}")
        record(True, f"{model_name} batch matches individual inference and both zero/NaN pairs", "")
    except Exception as e:
        record(False, "", f"{model_name} mixed-batch checks failed: {e}")

try:
    pd.testing.assert_frame_equal(batch_df, batch_before)
    if prepared_batch is not None:
        pd.testing.assert_frame_equal(prepared_batch, expected_batch)
    record(True, "Batch preprocessing and inference do not mutate original or prepared frames", "")
except AssertionError as e:
    record(False, "", f"Batch preprocessing or inference mutated a frame: {e}")

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Checks run: {checks_run}")
print(f"Checks passed: {checks_passed}")
print(f"Checks failed: {len(failures)}")

if failures:
    print("\nRequired tests failed:")
    for item in failures:
        print(f"  - {item}")
    print("=" * 80)
    sys.exit(1)

print("All tests passed!")
print("Ready to run: python -m streamlit run app.py")
print("=" * 80)
