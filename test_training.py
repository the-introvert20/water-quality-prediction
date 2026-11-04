"""Integrity tests for the training and evaluation pipeline.

Run with ``python test_training.py`` (exits non-zero on any failure). The first group uses
small synthetic data and runs in about a minute; the second group checks the artifacts
written by ``python train_model.py``.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import train_model as tm
from model_components import FoldBalancedXGBClassifier, PotableThresholdClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS = BASE_DIR / 'models'
DATASET = BASE_DIR / 'dataset' / 'water_potability.csv'


def synthetic_data(rows=360, seed=0):
    features, target = make_classification(n_samples=rows, n_features=9, n_informative=4, n_redundant=0,
                                           weights=[.6, .4], class_sep=.6, flip_y=.1, random_state=seed)
    frame = pd.DataFrame(features, columns=tm.FEATURES)
    rng = np.random.default_rng(seed)
    for column in ['ph', 'Sulfate', 'Trihalomethanes']:
        frame.loc[rng.random(rows) < .15, column] = np.nan
    return frame, pd.Series(target, name='Potability')


class StubProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, potable):
        self.potable = np.asarray(potable)

    def predict_proba(self, X):
        return np.column_stack([1 - self.potable, self.potable])

    def predict(self, X):
        return (self.potable > .5).astype(int)


class LeakageSafeComponentTests(unittest.TestCase):
    def test_xgboost_class_ratio_uses_only_fit_labels(self):
        X = np.random.default_rng(1).normal(size=(40, 3))
        model = FoldBalancedXGBClassifier(n_estimators=3, random_state=0)
        model.fit(X, np.array([0] * 30 + [1] * 10))
        self.assertAlmostEqual(model.get_params()['scale_pos_weight'], 3.0)
        # Refitting on different labels must replace the ratio, not keep a stale full-data value.
        model.fit(X, np.array([0] * 20 + [1] * 20))
        self.assertAlmostEqual(model.get_params()['scale_pos_weight'], 1.0)
        with self.assertRaises(ValueError):
            model.fit(X, np.zeros(40, dtype=int))

    def test_every_candidate_embeds_preprocessing_and_fits_it_on_training_rows_only(self):
        X, y = synthetic_data()
        X_train, X_other = X.iloc[:240], X.iloc[240:] + 1000.0   # wildly different "validation" rows
        y_train = y.iloc[:240]
        expected_medians = X_train.median().to_numpy()
        params = {'Random_Forest': {'max_depth': 3}, 'SVC': {'C': 1}, 'Extra_Trees': {'min_samples_leaf': 5}}
        for name in tm.MODEL_ORDER:
            with self.subTest(model=name):
                chosen = params if name == 'Soft_Voting_RF_SVC' else params.get(name, {})
                pipeline = tm.make_pipeline(name, chosen, tm.FAST_CONFIG).fit(X_train, y_train)
                self.assertEqual(list(pipeline.named_steps), ['imputer', 'scaler', 'classifier'])
                np.testing.assert_allclose(pipeline.named_steps['imputer'].statistics_, expected_medians)
                np.testing.assert_allclose(pipeline.named_steps['scaler'].mean_,
                                           X_train.fillna(X_train.median()).mean().to_numpy())
                self.assertEqual(list(pipeline.feature_names_in_), tm.FEATURES)
                pipeline.predict(X_other)   # prediction must not refit preprocessing
                np.testing.assert_allclose(pipeline.named_steps['imputer'].statistics_, expected_medians)

    def test_no_resampling_and_no_engineered_features(self):
        source = (BASE_DIR / 'train_model.py').read_text(encoding='utf-8')
        code = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('#'))
        for marker in ['imblearn', 'RandomOverSampler', 'RandomUnderSampler', 'ADASYN', 'resample(']:
            self.assertNotIn(marker, code)
        self.assertNotIn('import SMOTE', code)
        self.assertEqual(tm.FEATURES, ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

    def test_threshold_wrapper_applies_cut_off_and_keeps_probabilities(self):
        potable = np.array([.2, .55, .65, .9])
        stub = StubProbabilityModel(potable)
        for threshold, expected in [(None, [0, 1, 1, 1]), (.5, [0, 1, 1, 1]), (.6, [0, 0, 1, 1]), (.95, [0, 0, 0, 0])]:
            wrapper = PotableThresholdClassifier(stub, threshold)
            wrapper.estimator_, wrapper.classes_ = stub, stub.classes_
            np.testing.assert_array_equal(wrapper.predict(None), expected)
            np.testing.assert_allclose(wrapper.predict_proba(None)[:, 1], potable)


class ThresholdAndStatisticsTests(unittest.TestCase):
    def test_threshold_is_lowest_value_meeting_target_precision(self):
        y = np.array([0] * 60 + [1] * 60)
        proba = np.concatenate([np.linspace(.05, .65, 60), np.linspace(.35, .99, 60)])
        choice = tm.select_threshold(y, proba, target=.7, min_predicted=10)
        self.assertTrue(choice['target_met'])
        curve = {row['threshold']: row for row in choice['curve']}
        self.assertGreaterEqual(curve[choice['threshold']]['precision_potable'], .7)
        for row in choice['curve']:
            if row['threshold'] < choice['threshold'] and row['n_predicted_potable'] >= 10:
                self.assertLess(row['precision_potable'], .7)

    def test_threshold_falls_back_to_native_rule_when_target_unreachable_or_support_small(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        noise = rng.random(400)
        self.assertIsNone(tm.select_threshold(y, noise, target=.95, min_predicted=10)['threshold'])
        y2 = np.array([0] * 100 + [1] * 3)
        proba2 = np.array([.1] * 100 + [.95] * 3)
        self.assertIsNone(tm.select_threshold(y2, proba2, target=.7, min_predicted=50)['threshold'])

    def test_threshold_selection_depends_only_on_supplied_labels_and_scores(self):
        y = np.array([0, 1] * 100)
        proba = np.random.default_rng(3).random(200)
        first = tm.select_threshold(y, proba, target=.5, min_predicted=5)
        second = tm.select_threshold(y.copy(), proba.copy(), target=.5, min_predicted=5)
        self.assertEqual(first['threshold'], second['threshold'])

    def test_wilson_interval_and_corrected_ttest(self):
        low, high = tm.wilson_interval(50, 100)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertEqual(tm.wilson_interval(0, 0), [None, None])
        same = tm.corrected_paired_ttest([.6, .62, .61, .63, .6], [.6, .62, .61, .63, .6], 2096, 524)
        self.assertEqual(same['p_value'], 1.0)
        noisy = tm.corrected_paired_ttest([.62, .60, .64, .58, .63], [.61, .62, .60, .59, .60], 2096, 524)
        self.assertGreater(noisy['p_value'], .05)
        clear = tm.corrected_paired_ttest([.70, .71, .70, .72, .71], [.60, .61, .60, .61, .60], 2096, 524)
        self.assertLess(clear['p_value'], .05)


class ExperimentIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = synthetic_data(rows=420)
        cls.split = train_test_split(X, y, test_size=.25, stratify=y, random_state=7)

    def run_experiment(self, flip_test_labels):
        X_train, X_test, y_train, y_test = self.split
        if flip_test_labels:
            y_test = 1 - y_test
        return tm.run_experiment(X_train, X_test, y_train, y_test, tm.FAST_CONFIG)

    def test_test_labels_never_influence_selection_tuning_or_thresholds(self):
        pipelines_a, results_a, summary_a = self.run_experiment(False)
        pipelines_b, results_b, summary_b = self.run_experiment(True)
        self.assertEqual(summary_a['selection']['selected_model'], summary_b['selection']['selected_model'])
        self.assertEqual(summary_a['nested_selected_params'], summary_b['nested_selected_params'])
        for name in tm.MODEL_ORDER:
            with self.subTest(model=name):
                for key in ['cv_macro_f1_mean', 'cv_accuracy_mean', 'train_accuracy_mean',
                            'decision_threshold', 'oof_threshold_precision_potable', 'hyperparameters']:
                    self.assertEqual(results_a[name][key], results_b[name][key], key)
        # The test metrics, by contrast, must respond to the test labels.
        self.assertNotEqual(results_a['Random_Forest']['test_accuracy'], results_b['Random_Forest']['test_accuracy'])
        self.assertEqual(len(results_a['Random_Forest']['cv_macro_f1_folds']), tm.FAST_CONFIG['outer_splits'])

    def test_outputs_are_consistent_and_xgboost_ratio_comes_from_training_labels(self):
        pipelines, results, summary = self.run_experiment(False)
        _, X_test, y_train, y_test = self.split
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        self.assertAlmostEqual(pipelines['XGBoost'][-1].estimator_.get_params()['scale_pos_weight'], ratio)
        best = summary['selection']['selected_model']
        self.assertEqual(best, max(tm.SAVED_MODELS, key=lambda n: results[n]['cv_macro_f1_mean']))
        for name in tm.MODEL_ORDER:
            r = results[name]
            cm = np.array(r['confusion_matrix'])
            self.assertEqual(int(cm.sum()), len(y_test))
            self.assertEqual(cm[0].sum(), int((y_test == 0).sum()))
            prediction = pipelines[name].predict(X_test)
            np.testing.assert_array_equal(cm, confusion_matrix(y_test, prediction, labels=[0, 1]))
            self.assertAlmostEqual(r['test_accuracy'], accuracy_score(y_test, prediction))
            self.assertAlmostEqual(r['test_macro_f1'], f1_score(y_test, prediction, average='macro'))
            self.assertIn('0 = Not Potable', r['test_classification_report'])
            threshold = r['decision_threshold']
            self.assertEqual(pipelines[name][-1].threshold, threshold)
            if threshold is not None:
                proba = pipelines[name].predict_proba(X_test)[:, 1]
                np.testing.assert_array_equal(prediction, (proba >= threshold).astype(int))
        report = tm.format_report(results, summary)
        for section in ['SELECTED MODEL', 'DECISION THRESHOLD', 'Classification report', 'Confusion Matrix',
                        'LIMITATIONS AND UNCERTAINTY', 'DO NOT use as sole certification', 'Overfit Gap']:
            self.assertIn(section, report)


@unittest.skipUnless((MODELS / 'training_summary.json').exists(), 'run python train_model.py first')
class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((MODELS / 'evaluation_results.json').read_text(encoding='utf-8'))
        cls.summary = json.loads((MODELS / 'training_summary.json').read_text(encoding='utf-8'))
        cls.baseline = json.loads((MODELS / 'baseline_evaluation_results.json').read_text(encoding='utf-8'))
        data = pd.read_csv(DATASET)
        cls.data = data
        X = tm.treat_zero_ph_as_missing(data[tm.FEATURES])
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, data.Potability, test_size=.2, random_state=42, stratify=data.Potability)

    def test_original_dataset_is_unchanged_since_training(self):
        self.assertEqual(hashlib.sha256(DATASET.read_bytes()).hexdigest(), self.summary['dataset']['sha256'])
        self.assertEqual(self.summary['dataset']['rows'], len(self.data))

    def test_original_evaluation_is_preserved_as_baseline(self):
        rf = self.baseline['Random_Forest']
        self.assertEqual(rf['test_accuracy'], 0.6570121951219512)
        self.assertEqual(rf['test_macro_f1'], 0.6245106045226629)
        self.assertEqual(rf['confusion_matrix'], [[312, 88], [137, 119]])
        self.assertEqual(self.summary['before_after'][0]['before_test_accuracy'], rf['test_accuracy'])

    def test_all_candidates_reported_with_consistent_metrics(self):
        self.assertEqual(sorted(self.results), sorted(tm.MODEL_ORDER))
        self.assertEqual(self.summary['split']['test_size'], len(self.y_test))
        for name, r in self.results.items():
            with self.subTest(model=name):
                cm = np.array(r['confusion_matrix'])
                self.assertEqual(int(cm.sum()), len(self.y_test))
                self.assertAlmostEqual(r['test_accuracy'], np.trace(cm) / cm.sum())
                self.assertAlmostEqual(r['test_recall_class_1'], cm[1, 1] / cm[1].sum())
                if cm[:, 1].sum():
                    self.assertAlmostEqual(r['test_precision_class_1'], cm[1, 1] / cm[:, 1].sum())
                self.assertAlmostEqual(r['test_macro_f1'], (r['test_f1_class_0'] + r['test_f1_class_1']) / 2)
                self.assertEqual(r['actual_not_potable_pred_potable'], int(cm[0, 1]))
                self.assertEqual(len(r['cv_macro_f1_folds']), 5)
                self.assertAlmostEqual(r['cv_macro_f1_mean'], np.mean(r['cv_macro_f1_folds']))
                self.assertAlmostEqual(r['cv_macro_f1_std'], np.std(r['cv_macro_f1_folds']))
                self.assertIn('Not Potable', r['test_classification_report'])

    def test_selection_uses_validation_scores_and_is_justified(self):
        selected = self.summary['selection']['selected_model']
        self.assertNotEqual(selected, tm.BASELINE)
        self.assertEqual(selected, max(tm.SAVED_MODELS, key=lambda n: self.results[n]['cv_macro_f1_mean']))
        self.assertTrue(self.summary['selection']['justification'])
        self.assertEqual(len(self.summary['selection']['paired_comparisons_vs_selected']), len(tm.SAVED_MODELS) - 1)

    def test_thresholds_were_chosen_from_training_out_of_fold_curves(self):
        policy = self.summary['threshold_policy']
        for name in tm.SAVED_MODELS:
            with self.subTest(model=name):
                r = self.results[name]
                curve = r['oof_threshold_curve']
                qualifying = [row['threshold'] for row in curve
                              if row['n_predicted_potable'] >= policy['min_predicted_potable']
                              and row['precision_potable'] >= policy['target_potable_precision']]
                self.assertEqual(r['decision_threshold'], qualifying[0] if qualifying else None)
                self.assertEqual(r['threshold_target_met'], bool(qualifying))
                # Counts must be internally consistent and fit the TRAINING partition's class totals.
                counts = self.summary['split']['train_class_counts']
                for row in curve:
                    true_potable = round(row['recall_potable'] * counts['1'])
                    self.assertAlmostEqual(row['recall_potable'] * counts['1'], true_potable, places=6)
                    self.assertEqual(true_potable + row['false_not_potable'], counts['1'])
                    self.assertEqual(true_potable + row['false_potable'], row['n_predicted_potable'])
                    self.assertLessEqual(row['false_potable'], counts['0'])

    def test_saved_pipelines_match_reported_test_results(self):
        for name in tm.SAVED_MODELS:
            with self.subTest(model=name):
                pipeline = joblib.load(MODELS / f'{name}.pkl')
                self.assertEqual(list(pipeline.named_steps), ['imputer', 'scaler', 'classifier'])
                self.assertEqual(list(pipeline.feature_names_in_), tm.FEATURES)
                self.assertEqual(pipeline[-1].threshold, self.results[name]['decision_threshold'])
                prediction = pipeline.predict(self.X_test)
                self.assertAlmostEqual(accuracy_score(self.y_test, prediction), self.results[name]['test_accuracy'])
                self.assertEqual(confusion_matrix(self.y_test, prediction, labels=[0, 1]).tolist(),
                                 self.results[name]['confusion_matrix'])

    def test_saved_xgboost_uses_training_partition_class_ratio(self):
        pipeline = joblib.load(MODELS / 'XGBoost.pkl')
        training_ratio = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        full_ratio = (self.data.Potability == 0).sum() / (self.data.Potability == 1).sum()
        weight = pipeline[-1].estimator_.get_params()['scale_pos_weight']
        self.assertAlmostEqual(weight, training_ratio, places=9)
        self.assertNotAlmostEqual(weight, full_ratio, places=9)

    def test_report_covers_required_sections_and_keeps_safety_disclaimer(self):
        report = (MODELS / 'evaluation_report.txt').read_text(encoding='utf-8')
        for heading in ['SELECTED MODEL', 'NESTED CROSS-VALIDATION RESULTS', 'DECISION THRESHOLD',
                        'Classification report', 'Confusion Matrix', 'Overfit Gap',
                        'BEFORE / AFTER COMPARISON', 'LIMITATIONS AND UNCERTAINTY',
                        'DO NOT use as sole certification for drinking water safety']:
            self.assertIn(heading, report)
        for name in tm.MODEL_ORDER:
            self.assertIn(f'MODEL: {name}', report)
        self.assertNotRegex(report, re.compile(r'certif(?:ies|ied)\s+(?:safe|drinking)', re.I))


if __name__ == '__main__':
    unittest.main(verbosity=2)
