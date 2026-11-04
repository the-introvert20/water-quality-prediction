"""Final audit of the evaluation methodology (read-only; run from the project root).

Checks, using the real code in ``train_model.py``:
  1. threshold selection inside each outer training fold never sees outer validation labels
  2. model selection uses nested-CV Macro F1 only (no test data exists when it runs)
  3. probability calibration of SVC and the soft-voting ensemble on outer-fold held-out data
  4. exact derivation of the 0.758 +/- 0.036 nested Potable-precision estimate
  5. the test partition is evaluated once, after all selection, tuning and thresholding

The real test partition is never read here: the data split discards it and every
test-set check uses synthetic data. Outputs go to ``audit/``. Nothing under ``models/``,
``dataset/`` or the preserved baseline files is written.
"""
import ast
import inspect
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import train_model as tm
from test_training import synthetic_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)
CONFIG = tm.DEFAULT_CONFIG
AUDITED = ['SVC', 'Soft_Voting_RF_SVC']


def jsonable(value):
    return json.loads(json.dumps(tm.to_serializable(value)))


def calibration_summary(y, p, bins=10):
    """Brier, log loss, reliability by equal-frequency bins, ECE/MCE, calibration slope/intercept."""
    y, p = np.asarray(y), np.asarray(p)
    prevalence = y.mean()
    brier = brier_score_loss(y, p)
    reference = prevalence * (1 - prevalence)          # Brier of always predicting the base rate
    order = np.argsort(p, kind='stable')
    rows = []
    for index in np.array_split(order, bins):
        n = len(index)
        observed = int(y[index].sum())
        low, high = tm.wilson_interval(observed, n)
        rows.append({'n': n, 'p_min': float(p[index].min()), 'p_max': float(p[index].max()),
                     'mean_predicted': float(p[index].mean()), 'observed_rate': observed / n,
                     'observed_ci95': [low, high]})
    ece = sum(r['n'] * abs(r['observed_rate'] - r['mean_predicted']) for r in rows) / len(y)
    mce = max(abs(r['observed_rate'] - r['mean_predicted']) for r in rows)
    logit = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return {
        'n': int(len(y)), 'prevalence': float(prevalence), 'brier': float(brier),
        'brier_base_rate_reference': float(reference), 'brier_skill_score': float(1 - brier / reference),
        'log_loss': float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        'ece_equal_frequency_10_bins': float(ece), 'max_bin_gap': float(mce),
        'calibration_slope': float(fit.coef_[0, 0]), 'calibration_intercept': float(fit.intercept_[0]),
        'probability_range': [float(p.min()), float(p.max())], 'reliability_bins': rows,
    }


def threshold_region(y, p, threshold):
    predicted = p >= threshold
    n = int(predicted.sum())
    true_potable = int(y[predicted].sum())
    return {'threshold': threshold, 'n_predicted_potable': n, 'observed_potable_rate': true_potable / n,
            'mean_predicted_potable_probability': float(p[predicted].mean()),
            'observed_ci95': tm.wilson_interval(true_potable, n)}


def audit_nested_folds(X_train, y_train, stored, summary):
    outer = StratifiedKFold(CONFIG['outer_splits'], shuffle=True, random_state=tm.OUTER_SEED)
    spy_calls = []
    original_select = tm.select_threshold

    def spy(y_true, proba, *args, **kwargs):
        spy_calls.append((np.asarray(y_true).copy(), np.asarray(proba).copy()))
        return original_select(y_true, proba, *args, **kwargs)

    tm.select_threshold = spy
    folds = {name: [] for name in AUDITED}
    held_out = {name: {'y': [], 'p': [], 'fold': []} for name in AUDITED}
    checks = []
    try:
        for fold, (train_index, val_index) in enumerate(outer.split(X_train, y_train), start=1):
            X_fold, y_fold = X_train.iloc[train_index], y_train.iloc[train_index]
            X_val, y_val = X_train.iloc[val_index], y_train.iloc[val_index]
            chosen = {}
            for name in ['Random_Forest', 'SVC', 'Soft_Voting_RF_SVC']:
                chosen[name] = tm.tune_hyperparameters(name, X_fold, y_fold, CONFIG, chosen)
            for name in AUDITED:
                spy_calls.clear()
                result = tm.evaluate_fold(name, chosen[name], X_fold, y_fold, X_val, y_val, CONFIG)
                assert len(spy_calls) == 1, 'threshold selection must run exactly once per fold'
                seen_labels, seen_scores = spy_calls[0]
                same_labels = (len(seen_labels) == len(y_fold)
                               and np.array_equal(seen_labels, y_fold.to_numpy()))
                # Labels seen are the outer TRAINING labels only; none belong to the outer validation fold.
                checks.append({'fold': fold, 'model': name, 'labels_seen': int(len(seen_labels)),
                               'outer_train_rows': int(len(y_fold)), 'outer_validation_rows': int(len(y_val)),
                               'labels_identical_to_outer_training_labels': bool(same_labels),
                               'scores_length_equals_outer_training_rows': bool(len(seen_scores) == len(y_fold))})
                assert same_labels
                # Counterfactual: corrupt the outer validation labels; the threshold must not move.
                flipped = tm.evaluate_fold(name, chosen[name], X_fold, y_fold, X_val, 1 - y_val, CONFIG)
                checks[-1].update({
                    'threshold': result['thr_threshold'],
                    'threshold_with_flipped_validation_labels': flipped['thr_threshold'],
                    'threshold_unchanged': result['thr_threshold'] == flipped['thr_threshold'],
                    'validation_precision_changes_with_labels': result['thr_precision_1'] != flipped['thr_precision_1'],
                })
                assert result['thr_threshold'] == flipped['thr_threshold']
                folds[name].append(result)
                pipeline = tm.make_pipeline(name, chosen[name], CONFIG).fit(X_fold, y_fold)
                held_out[name]['y'].append(y_val.to_numpy())
                held_out[name]['p'].append(pipeline.predict_proba(X_val)[:, 1])
                held_out[name]['fold'].append(np.full(len(y_val), fold))
            print(f'  audited outer fold {fold}/{CONFIG["outer_splits"]}', flush=True)
            for name in AUDITED:
                expected = summary['nested_selected_params'][name][fold - 1]
                assert jsonable(chosen[name]) == expected, (name, fold, chosen[name], expected)
    finally:
        tm.select_threshold = original_select

    replication = {}
    for name in AUDITED:
        aggregate = tm.aggregate_folds(folds[name])
        for key in ['cv_macro_f1_mean', 'cv_macro_f1_std', 'cv_threshold_precision_1_mean',
                    'cv_threshold_precision_1_std', 'cv_threshold_recall_1_mean']:
            np.testing.assert_allclose(aggregate[key], stored[name][key], rtol=0, atol=1e-12)
        assert aggregate['cv_threshold_values'] == stored[name]['cv_threshold_values']
        replication[name] = 'exactly reproduces the stored nested-CV results and selected parameters'
    return folds, held_out, checks, replication


def precision_derivation(folds, name):
    rows = []
    for index, fold in enumerate(folds[name], start=1):
        n = fold['thr_n_predicted_potable']
        false_potable = fold['thr_false_potable']
        true_potable = n - false_potable
        rows.append({'fold': index, 'threshold_chosen_in_fold': fold['thr_threshold'],
                     'n_predicted_potable': n, 'true_potable': true_potable, 'false_potable': false_potable,
                     'precision': fold['thr_precision_1'], 'recall': fold['thr_recall_1']})
        assert abs(true_potable / n - fold['thr_precision_1']) < 1e-12
    precision = np.array([row['precision'] for row in rows])
    n_total = sum(row['n_predicted_potable'] for row in rows)
    tp_total = sum(row['true_potable'] for row in rows)
    k = len(rows)
    sd0, sd1 = float(precision.std()), float(precision.std(ddof=1))
    margin = float(stats.t.ppf(.975, k - 1) * sd1 / math.sqrt(k))
    return {
        'per_fold': rows, 'mean_of_fold_precisions': float(precision.mean()),
        'sd_ddof0_as_reported': sd0, 'sd_ddof1': sd1,
        't95_interval_for_mean_precision': [float(precision.mean() - margin), float(precision.mean() + margin)],
        'pooled_precision': tp_total / n_total, 'pooled_true_potable': tp_total, 'pooled_predicted_potable': n_total,
        'pooled_wilson_95': tm.wilson_interval(tp_total, n_total),
        'pooled_false_potable': n_total - tp_total,
    }


def final_threshold_reproduction(X_train, y_train, stored):
    outer = StratifiedKFold(CONFIG['outer_splits'], shuffle=True, random_state=tm.OUTER_SEED)
    out = {}
    for name in AUDITED:
        params = stored[name]['hyperparameters']
        oof = cross_val_predict(tm.make_pipeline(name, params, CONFIG), X_train, y_train, cv=outer,
                                method='predict_proba', n_jobs=-1)[:, 1]
        choice = tm.select_threshold(y_train, oof, CONFIG['target_precision'], CONFIG['min_predicted_potable'])
        out[name] = {'recomputed_threshold': choice['threshold'], 'stored_threshold': stored[name]['decision_threshold'],
                     'identical_curve': jsonable(choice['curve']) == stored[name]['oof_threshold_curve'],
                     'oof_labels_are_training_labels_only': len(oof) == len(y_train)}
        assert out[name]['recomputed_threshold'] == out[name]['stored_threshold']
        assert out[name]['identical_curve']
    return out


def selection_audit(stored, summary):
    names = tm.SAVED_MODELS
    recomputed = {name: float(np.mean(summary['nested_fold_scores'][name]['macro_f1'])) for name in names}
    ranking = sorted(names, key=lambda n: recomputed[n], reverse=True)
    source = inspect.getsource(tm.run_experiment)
    selection_at = source.index('select_best_model(')
    final_fit_at = source.index('fit_final_pipelines(')
    test_eval_at = source.index('evaluate_on_test_set(')
    selection_source = inspect.getsource(tm.select_best_model)
    margin = recomputed[ranking[0]] - recomputed[ranking[1]]
    deployed = {name: stored[name]['cv_threshold_macro_f1_mean'] for name in ['Random_Forest', 'SVC', 'Extra_Trees', 'Soft_Voting_RF_SVC']}
    return {
        'recomputed_nested_macro_f1_from_fold_scores': recomputed, 'ranking': ranking,
        'selected_in_summary': summary['selection']['selected_model'],
        'recomputed_winner_matches_selected': ranking[0] == summary['selection']['selected_model'],
        'margin_over_runner_up': margin,
        'winner_and_runner_up': ranking[:2],
        'select_best_model_called_before_test_evaluation': selection_at < final_fit_at < test_eval_at,
        'select_best_model_source_mentions_test_data': any(t in selection_source for t in ['X_test', 'y_test', 'test_']),
        'nested_threshold_policy_macro_f1_for_tuned_models': deployed,
        'selected_model_best_on_threshold_policy_macro_f1': max(deployed, key=deployed.get) == summary['selection']['selected_model'],
        'fold_sd_selected_vs_runner_up': [float(np.std(summary['nested_fold_scores'][n]['macro_f1'])) for n in ranking[:2]],
        'train_validation_gap_selected_vs_runner_up': [stored[n]['train_accuracy_mean'] - stored[n]['cv_accuracy_mean'] for n in ranking[:2]],
    }


def test_partition_audit():
    tree = ast.parse(Path(tm.__file__).read_text(encoding='utf-8'))
    users = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in {'X_test', 'y_test'}}
            names |= {a.arg for a in node.args.args if a.arg in {'X_test', 'y_test'}}
            if names:
                users[node.name] = sorted(names)

    X, y = synthetic_data(rows=420)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=7)
    rng = np.random.default_rng(99)
    other_X = pd.DataFrame(rng.normal(size=X_test.shape), columns=X_test.columns, index=X_test.index)
    other_y = pd.Series(rng.integers(0, 2, len(y_test)), index=y_test.index, name='Potability')
    log, selection_inputs = [], []
    originals = {name: getattr(tm, name) for name in
                 ['nested_cross_validation', 'select_threshold', 'select_best_model',
                  'fit_final_pipelines', 'evaluate_on_test_set']}

    def wrap(name):
        def inner(*args, **kwargs):
            log.append(name)
            if name == 'select_best_model':
                selection_inputs.append(any(k.startswith('test_') for r in args[0].values() for k in r))
            return originals[name](*args, **kwargs)
        setattr(tm, name, inner)

    for name in originals:
        wrap(name)
    try:
        _, results_a, summary_a = tm.run_experiment(X_train, X_test, y_train, y_test, tm.FAST_CONFIG)
        first_log = list(log)
        log.clear()
        _, results_b, summary_b = tm.run_experiment(X_train, other_X, y_train, other_y, tm.FAST_CONFIG)
    finally:
        for name, function in originals.items():
            setattr(tm, name, function)
    first_eval = first_log.index('evaluate_on_test_set')
    last_pre_test = max(i for i, n in enumerate(first_log) if n != 'evaluate_on_test_set')
    training_keys = ['cv_macro_f1_mean', 'cv_macro_f1_std', 'cv_accuracy_mean', 'train_accuracy_mean',
                     'decision_threshold', 'hyperparameters', 'oof_threshold_precision_potable',
                     'cv_threshold_precision_1_mean']
    identical = all(results_a[n].get(k) == results_b[n].get(k) for n in tm.MODEL_ORDER
                    for k in training_keys if k in results_a[n])
    return {
        'functions_that_reference_test_partition': users,
        'call_sequence_summary': {'evaluate_on_test_set_calls': first_log.count('evaluate_on_test_set'),
                                  'models': len(tm.MODEL_ORDER),
                                  'every_other_stage_precedes_first_test_evaluation': last_pre_test < first_eval,
                                  'select_best_model_calls': first_log.count('select_best_model'),
                                  'test_keys_present_when_selection_ran': selection_inputs[0]},
        'completely_different_test_features_and_labels': {
            'selection_unchanged': summary_a['selection']['selected_model'] == summary_b['selection']['selected_model'],
            'training_derived_results_identical': identical,
            'test_metrics_differ': results_a['Random_Forest']['test_accuracy'] != results_b['Random_Forest']['test_accuracy']},
    }


def plot_reliability(held_out_summaries, path):
    figure, (axis, hist) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.35, 1]})
    axis.plot([0, 1], [0, 1], '--', color='#6C8797', label='Perfect calibration')
    colors = {'SVC': '#C28B45', 'Soft_Voting_RF_SVC': '#08949C'}
    for name, summary in held_out_summaries.items():
        bins = summary['reliability_bins']
        axis.errorbar([b['mean_predicted'] for b in bins], [b['observed_rate'] for b in bins],
                      yerr=[[b['observed_rate'] - b['observed_ci95'][0] for b in bins],
                            [b['observed_ci95'][1] - b['observed_rate'] for b in bins]],
                      marker='o', capsize=3, color=colors[name], label=tm.DISPLAY_NAMES[name])
    axis.set_xlabel('Mean predicted P(Potable) per equal-frequency bin')
    axis.set_ylabel('Observed share labelled Potable (95% Wilson CI)')
    axis.set_title('Reliability, outer-fold held-out predictions (n = 2,620)')
    axis.legend(frameon=False)
    for name, values in held_out_summaries.items():
        hist.hist(values['_p'], bins=25, alpha=.6, color=colors[name], label=tm.DISPLAY_NAMES[name])
    hist.axvline(.55, color='#29485D', linestyle=':', label='Deployed threshold (ensemble) 0.55')
    hist.set_xlabel('Predicted P(Potable)')
    hist.set_ylabel('Records')
    hist.set_title('Distribution of predicted probabilities')
    hist.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    stored = json.loads((ROOT / 'models' / 'evaluation_results.json').read_text(encoding='utf-8'))
    summary = json.loads((ROOT / 'models' / 'training_summary.json').read_text(encoding='utf-8'))
    data = pd.read_csv(ROOT / 'dataset' / 'water_potability.csv')
    X = tm.treat_zero_ph_as_missing(data[tm.FEATURES])
    # The test partition is discarded immediately and never inspected by this audit.
    X_train, _, y_train, _ = train_test_split(X, data.Potability, test_size=.2,
                                              random_state=tm.RANDOM_STATE, stratify=data.Potability)
    report = {}
    print('[1/5] Re-running the audited nested folds with instrumentation ...', flush=True)
    folds, held_out, checks, replication = audit_nested_folds(X_train, y_train, stored, summary)
    report['threshold_independence'] = {'per_fold_checks': checks, 'replication': replication}
    print('[2/5] Selection audit ...', flush=True)
    report['selection'] = selection_audit(stored, summary)
    print('[3/5] Calibration on outer-fold held-out probabilities ...', flush=True)
    calibration, plot_data = {}, {}
    for name in AUDITED:
        y = np.concatenate(held_out[name]['y'])
        p = np.concatenate(held_out[name]['p'])
        per_fold_brier = [brier_score_loss(a, b) for a, b in zip(held_out[name]['y'], held_out[name]['p'])]
        calibration[name] = calibration_summary(y, p)
        calibration[name]['brier_per_fold_mean'] = float(np.mean(per_fold_brier))
        calibration[name]['brier_per_fold_std'] = float(np.std(per_fold_brier))
        deployed_threshold = stored[name]['decision_threshold']
        calibration[name]['at_deployed_threshold'] = threshold_region(y, p, deployed_threshold)
        calibration[name]['at_threshold_0.50'] = threshold_region(y, p, .5)
        plot_data[name] = {**calibration[name], '_p': p}
    plot_reliability(plot_data, OUT / 'calibration_reliability.png')
    report['calibration'] = calibration
    print('[4/5] Nested Potable-precision derivation ...', flush=True)
    report['precision_derivation'] = {name: precision_derivation(folds, name) for name in AUDITED}
    print('[5/5] Final-threshold reproduction and test-partition audit ...', flush=True)
    report['final_threshold_reproduction'] = final_threshold_reproduction(X_train, y_train, stored)
    report['test_partition'] = test_partition_audit()
    (OUT / 'audit_results.json').write_text(json.dumps(jsonable(report), indent=2), encoding='utf-8')
    print('AUDIT COMPLETE: all assertions held. Wrote audit/audit_results.json and audit/calibration_reliability.png')


if __name__ == '__main__':
    main()
