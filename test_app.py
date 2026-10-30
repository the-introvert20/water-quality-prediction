"""Run with python test_app.py; no browser or extra test dependencies required."""
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from water_quality import (
    BASE_DIR, CLASS_LABELS, DISCLAIMER, FEATURES, FIELD_SPECS, MODEL_NAMES,
    batch_results, load_pipeline, predict_frame, prediction_report, read_batch_csv,
    validate_features, reference_checks,
)


def sample_frame():
    return pd.DataFrame([{name: spec[3] for name, spec in FIELD_SPECS.items()}], columns=FEATURES)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {name: load_pipeline(name) for name in MODEL_NAMES}

    def test_numeric_schema_and_copy(self):
        frame = sample_frame()
        frame['ph'] = 0.0
        frame['sample_id'] = 'A-01'
        original = frame.copy(deep=True)
        prepared = validate_features(frame)
        self.assertTrue(pd.isna(prepared.ph.iloc[0]))
        self.assertEqual(list(prepared.columns), FEATURES)
        assert_frame_equal(frame, original)

    def test_batch_validation(self):
        valid = sample_frame().to_csv(index=False).encode()
        self.assertEqual(len(read_batch_csv(valid)), 1)
        self.assertEqual(len(read_batch_csv(b'\xef\xbb\xbf' + valid)), 1)
        bad_files = [b'', b'ph\n7\n', ','.join(FEATURES).encode(),
                     b'ph,ph\n7,8\n', b'\xff\xfe\x00',
                     valid.replace(b'7.0', b'invalid', 1),
                     valid.replace(b'7.0', b'inf', 1),
                                          valid + b'7,200,extra\n', valid.replace(b'7.0', b'"unterminated', 1)]
        for content in bad_files:
            with self.subTest(content=content[:35]), self.assertRaises(ValueError):
                read_batch_csv(content)

    def test_uploaded_identifiers_and_na_literals_are_preserved(self):
        frame = sample_frame()
        frame['sample_id'] = '00123'
        frame['site'] = 'NA'
        parsed = read_batch_csv(frame.to_csv(index=False).encode())
        self.assertEqual(parsed.sample_id.iloc[0], '00123')
        self.assertEqual(parsed.site.iloc[0], 'NA')
        result, _ = batch_results(self.models['Random_Forest'], parsed, 'Random_Forest')
        restored = pd.read_csv(io.BytesIO(result.to_csv(index=False).encode()), dtype=str, keep_default_na=False)
        self.assertEqual(restored.sample_id.iloc[0], '00123')
        self.assertEqual(restored.site.iloc[0], 'NA')

    def test_invalid_cells(self):
        for bad in [np.inf, -np.inf, 'not-a-number']:
            frame = sample_frame().astype(object)
            frame.loc[0, 'ph'] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'ph:'):
                validate_features(frame)

    def test_missing_numeric_strings_and_blank_cells(self):
        frame = sample_frame().astype(str)
        frame.loc[0, 'ph'] = ' '
        prepared = validate_features(frame)
        self.assertTrue(pd.isna(prepared.ph.iloc[0]))
        self.assertEqual(prepared.Hardness.iloc[0], 196.0)

    def test_result_column_collision(self):
        frame = sample_frame()
        frame['Prediction'] = 'original'
        with self.assertRaisesRegex(ValueError, 'Rename existing result columns'):
            read_batch_csv(frame.to_csv(index=False).encode())
        with self.assertRaises(ValueError):
            batch_results(self.models['Random_Forest'], frame, 'Random_Forest')

    def test_manual_batch_missing_zero_consistency_all_models(self):
        frame = pd.concat([sample_frame()] * 3, ignore_index=True)
        frame.loc[0, 'ph'] = 0
        frame.loc[1, 'ph'] = np.nan
        frame['sample_id'] = ['zero', 'missing', 'complete']
        original = frame.copy(deep=True)
        for name, model in self.models.items():
            with self.subTest(model=name):
                results, prepared = batch_results(model, frame, name)
                self.assertEqual(results.Predicted_Class.iloc[0], results.Predicted_Class.iloc[1])
                self.assertTrue(prepared.ph.iloc[:2].isna().all())
                np.testing.assert_allclose(results.Prob_Potable.iloc[0], results.Prob_Potable.iloc[1])
                for position in range(3):
                    prediction, probability, _ = predict_frame(model, frame.iloc[[position]])
                    self.assertEqual(int(prediction[0]), results.Predicted_Class.iloc[position])
                    self.assertAlmostEqual(probability[0, 1] * 100, results.Prob_Potable.iloc[position])
                restored = pd.read_csv(io.BytesIO(results.to_csv(index=False).encode()))
                self.assertEqual(len(restored), 3)
                self.assertEqual(restored.sample_id.tolist(), frame.sample_id.tolist())
                self.assertEqual(restored.Prediction_Model.unique().tolist(), [name])
                assert_frame_equal(frame, original)

    def test_probability_order_uses_class_ids(self):
        class ReversedModel:
            classes_ = np.array([1, 0])

            def predict(self, frame):
                return np.ones(len(frame), dtype=int)

            def predict_proba(self, frame):
                return np.tile([.8, .2], (len(frame), 1))

        _, probability, _ = predict_frame(ReversedModel(), sample_frame())
        np.testing.assert_allclose(probability, [[.2, .8]])

    def test_model_without_probabilities(self):
        class PredictionOnly:
            def predict(self, frame):
                return np.zeros(len(frame), dtype=int)

        prediction, probability, _ = predict_frame(PredictionOnly(), sample_frame())
        self.assertIsNone(probability)
        report = prediction_report(sample_frame(), prediction[0], probability, 'Prediction only')
        self.assertIn(DISCLAIMER, report)
        results, _ = batch_results(PredictionOnly(), sample_frame(), 'Prediction only')
        self.assertNotIn('Prob_Potable', results)

    def test_legacy_reference_checks_are_not_certifications(self):
        frame = sample_frame()
        frame['ph'] = 0.0
        flags = reference_checks(frame)
        self.assertFalse(any('pH' in flag for flag in flags))
        self.assertTrue(any('Chloramines' in flag for flag in flags))
        frame['ph'] = 12.0
        self.assertTrue(any('pH' in flag for flag in reference_checks(frame)))
        report = prediction_report(frame, 0, None, 'Random_Forest')
        self.assertIn('not a standards-compliance assessment', report)

    def test_report_matches_actual_outputs(self):
        model = self.models['Random_Forest']
        frame = sample_frame()
        prediction, probability, _ = predict_frame(model, frame)
        report = prediction_report(frame, prediction[0], probability[0], 'Random_Forest')
        self.assertIn(CLASS_LABELS[int(prediction[0])], report)
        self.assertIn(DISCLAIMER, report)
        self.assertIn(f'{probability[0, 1]:.2%}', report)

    def test_saved_test_metrics_match_current_artifacts(self):
        data = pd.read_csv(BASE_DIR / 'dataset' / 'water_potability.csv')
        _, test, _, target = train_test_split(data[FEATURES], data.Potability,
                                              test_size=.2, random_state=42, stratify=data.Potability)
        evaluations = json.loads((BASE_DIR / 'models' / 'evaluation_results.json').read_text())
        for name, model in self.models.items():
            with self.subTest(model=name):
                prediction, _, _ = predict_frame(model, test)
                self.assertAlmostEqual(accuracy_score(target, prediction), evaluations[name]['test_accuracy'])
                self.assertAlmostEqual(f1_score(target, prediction, average='macro'), evaluations[name]['test_macro_f1'])
                self.assertAlmostEqual(f1_score(target, prediction, average='weighted'), evaluations[name]['test_weighted_f1'])
                self.assertEqual(confusion_matrix(target, prediction, labels=[0, 1]).tolist(), evaluations[name]['confusion_matrix'])


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(BASE_DIR / 'app.py'), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def navigate(self, view):
        self.app.radio(key='workspace').set_value(view).run()
        self.assertFalse(self.app.exception)

    def click(self, label):
        button = next(button for button in self.app.button if button.label == label)
        button.click().run()
        self.assertFalse(self.app.exception)

    def test_all_views_and_charts(self):
        for view, minimum_charts in [('Overview', 1), ('Dataset Explorer', 4),
                                      ('Model Performance', 4), ('Training & Methodology', 0),
                                      ('Batch Analysis', 0), ('Water Prediction', 0)]:
            with self.subTest(view=view):
                self.navigate(view)
                self.assertGreaterEqual(len(self.app.get('plotly_chart')), minimum_charts)
                self.assertTrue(any(DISCLAIMER in item.value for item in list(self.app.info) + list(self.app.warning)))

    def test_manual_prediction_and_report_persist(self):
        self.navigate('Water Prediction')
        self.click('Estimate potability')
        self.assertIn('manual_result', self.app.session_state)
        self.assertEqual(len(self.app.get('download_button')), 1)
        self.assertEqual(len(self.app.get('plotly_chart')), 1)
        self.navigate('Overview')
        self.navigate('Water Prediction')
        self.assertEqual(len(self.app.get('download_button')), 1)

    def test_example_buttons_and_model_change(self):
        self.navigate('Water Prediction')
        for label, expected in [('Load Potable example', 1), ('Load Not Potable example', 0)]:
            self.click(label)
            self.click('Estimate potability')
            self.assertEqual(self.app.session_state.manual_result['prediction'], expected)
        self.click('Load random example')
        self.app.selectbox(key='selected_model').set_value('Logistic_Regression').run()
        self.assertFalse(self.app.exception)
        self.click('Estimate potability')
        self.assertEqual(self.app.session_state.manual_result['model'], 'Logistic_Regression')

    def test_zero_and_missing_ph_in_form(self):
        self.navigate('Water Prediction')
        self.app.number_input(key='measure_ph').set_value(0.0).run()
        self.click('Estimate potability')
        zero = self.app.session_state.manual_result['probabilities'].copy()
        self.app.checkbox(key='ph_missing').check().run()
        self.click('Estimate potability')
        missing = self.app.session_state.manual_result['probabilities']
        np.testing.assert_allclose(zero, missing)

    def test_batch_ui_valid_and_invalid(self):
        # AppTest has no file-uploader driver. Inject bytes through the uploader
        # boundary to test rendering/state; real browser upload is separate.
        uploaded = io.BytesIO(sample_frame().to_csv(index=False).encode())
        with patch('streamlit.file_uploader', return_value=uploaded):
            self.navigate('Batch Analysis')
            self.click('Analyze batch')
            self.assertEqual(len(self.app.session_state.batch_result[1]), 1)
            self.assertEqual(len(self.app.get('download_button')), 2)
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.app.get('download_button')), 2)
        with patch('streamlit.file_uploader', return_value=io.BytesIO(b'ph\n7\n')):
            self.app.run()
            self.assertTrue(any('Missing required columns' in error.value for error in self.app.error))

    def test_evaluation_fallback_keeps_app_accessible(self):
        self.app = AppTest.from_string(
            'import app\n'
            'from unittest.mock import patch\n'
            'with patch.object(app, "cached_evaluation", side_effect=ValueError("invalid JSON")):\n'
            '    app.main()\n', default_timeout=30).run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any('Evaluation report unavailable' in value.value for value in self.app.warning))
        self.assertEqual(self.app.metric[0].value, 'Unavailable')

    def test_saved_result_is_hidden_when_model_becomes_unavailable(self):
        self.app = AppTest.from_string(
            'import app\n'
            'from unittest.mock import patch\n'
            'if app.st.session_state.get("break_model", False):\n'
            '    with patch.object(app, "cached_pipeline", side_effect=ValueError("invalid artifact")):\n'
            '        app.main()\n'
            'else:\n'
            '    app.main()\n', default_timeout=30).run()
        self.navigate('Water Prediction')
        self.click('Estimate potability')
        self.assertEqual(len(self.app.get('download_button')), 1)
        self.app.session_state['break_model'] = True
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.get('download_button')), 0)

    def test_model_fallback_keeps_dataset_accessible(self):
        self.app = AppTest.from_string(
            'import app\n'
            'from unittest.mock import patch\n'
            'with patch.object(app, "cached_pipeline", side_effect=ValueError("invalid artifact")):\n'
            '    app.main()\n', default_timeout=30).run()
        self.assertFalse(self.app.exception)
        self.navigate('Dataset Explorer')
        self.assertGreaterEqual(len(self.app.get('plotly_chart')), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
