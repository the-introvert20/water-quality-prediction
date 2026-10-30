"""Water Quality Intelligence: a dataset-based educational ML dashboard."""
import hashlib
import json
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from water_quality import (
    BASE_DIR, CLASS_LABELS, CLASS_NAMES, DISCLAIMER, FEATURES, FIELD_SPECS,
    MODEL_NAMES, batch_results, feature_importance, load_pipeline,
    predict_frame, prediction_report, read_batch_csv, reference_checks,
)

st.set_page_config(page_title='Water Quality Intelligence',
                   page_icon=':material/water_drop:', layout='wide')

NAVIGATION = ['Overview', 'Water Prediction', 'Batch Analysis', 'Dataset Explorer',
              'Model Performance', 'Training & Methodology']
COLORS = {CLASS_LABELS[0]: '#29485D', CLASS_LABELS[1]: '#08949C'}

# Only owned presentation classes are styled; widgets use the native theme.
SHELL_CSS = """
<style>
.wqi-hero {background:#102D3D;color:#F5FAFC;padding:26px 30px;border-radius:8px;
 border-bottom:3px solid #169BA5;position:relative;overflow:hidden;margin-bottom:18px;}
.wqi-hero:after {content:'';position:absolute;width:200px;height:200px;border:1px solid #345466;
 border-radius:50%;right:-80px;top:-100px;pointer-events:none;}
.wqi-eyebrow {font-size:12px;letter-spacing:2px;font-weight:650;color:#8ED9DB;margin-bottom:9px;}
.wqi-hero h1 {font-size:clamp(25px,3vw,36px);font-weight:650;letter-spacing:-.8px;
 line-height:1.15;color:#F5FAFC;margin:0 0 10px;padding:0;}
.wqi-hero p {font-size:16px;line-height:1.6;color:#C6D9E2;max-width:780px;margin:0;}
.wqi-status {display:inline-block;margin-top:16px;font-size:13px;color:#BCE8E7;
 border:1px solid #3E6574;border-radius:4px;padding:5px 10px;}
.wqi-result {padding:22px 24px;border-left:4px solid #08949C;background:#EAF5F5;
 border-radius:4px;margin-bottom:16px;color:#132E40;}
.wqi-result h3 {font-size:28px;padding:0;margin:4px 0 8px;color:#132E40;}
.wqi-result p {margin:0;line-height:1.6;}
.wqi-kicker {font-size:12px;font-weight:650;letter-spacing:1.5px;text-transform:uppercase;}
.wqi-footer {border-top:1px solid #D9E4E9;margin-top:26px;padding-top:16px;
 color:#506B7B;font-size:13px;line-height:1.7;}
@media(max-width:640px) {.wqi-hero {padding:22px 20px;}.wqi-result {padding:18px;}}
</style>
"""


def artifact_version(path):
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


@st.cache_resource(max_entries=8, show_spinner=False)
def cached_pipeline(name, version, schema_version):
    return load_pipeline(name)


@st.cache_data(max_entries=2, show_spinner=False)
def cached_dataset(version):
    frame = pd.read_csv(BASE_DIR / 'dataset' / 'water_potability.csv')
    if set(FEATURES + ['Potability']) - set(frame.columns):
        raise ValueError('The dataset is missing required feature or target columns.')
    return frame


@st.cache_data(max_entries=2, show_spinner=False)
def cached_evaluation(version):
    report = json.loads((BASE_DIR / 'models' / 'evaluation_results.json').read_text(encoding='utf-8'))
    if not isinstance(report, dict):
        raise ValueError('Expected a model-keyed evaluation report.')
    for name, result in report.items():
        if not isinstance(result, dict):
            raise ValueError(f'Invalid evaluation entry: {name}')
    return report


def metric_value(result, key):
    value = result.get(key)
    if isinstance(value, (float, int)) and np.isfinite(value):
        return f'{value:.1%}'
    return 'Unavailable'


def metric_row(items):
    with st.container(horizontal=True, gap='small'):
        for label, value, help_text in items:
            if label == 'Selected model':
                with st.container(border=True, width=220, height=110):
                    st.markdown(label, help=help_text)
                    st.markdown(f'### {value}')
            else:
                st.metric(label, value, help=help_text, border=True, width=220, height=110)


def section(title, description):
    st.subheader(title)
    st.caption(description)


def chart(figure, key, height=340):
    figure.update_layout(
        template='plotly_white', height=height,
        font=dict(family='Arial, sans-serif', size=13, color='#29485D'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=12, r=16, t=52, b=24),
        title_font=dict(size=17), legend_title_text='',
        colorway=['#29485D', '#08949C', '#6C8797', '#C28B45'],
    )
    figure.update_xaxes(automargin=True, gridcolor='#E4ECEF', zeroline=False)
    figure.update_yaxes(automargin=True, gridcolor='#E4ECEF', zeroline=False)
    st.plotly_chart(figure, width='stretch', key=key, config={'displaylogo': False})


def distribution_chart(data):
    counts = data['Potability'].value_counts().reindex([0, 1], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=[CLASS_LABELS[0], CLASS_LABELS[1]], values=counts.values, hole=.72,
        sort=False, marker_colors=list(COLORS.values()), textinfo='percent',
        hovertemplate='%{label}<br>%{value:,} records · %{percent}<extra></extra>',
    ))
    fig.update_layout(title='Recorded target classes',
                      legend=dict(orientation='h', y=-.12, x=.5, xanchor='center'),
                      annotations=[dict(text=f'{len(data):,}<br>records', x=.5, y=.5,
                                        font_size=19, showarrow=False)])
    return fig


def overfit_notice(result):
    train, validation = result.get('train_accuracy_mean'), result.get('cv_accuracy_mean')
    if train is not None and validation is not None:
        gap = train - validation
        message = (f'Mean fold-training accuracy {train:.1%} versus validation accuracy '
                   f'{validation:.1%}: a {gap * 100:.1f} percentage-point gap. ')
        if gap > .15:
            st.warning(message + 'This large gap indicates overfitting; performance on new samples is substantially weaker.')
        else:
            st.info(message + 'A small gap alone does not establish real-world validity.')
    else:
        st.info('Training–validation comparison is unavailable in the saved report.')


def render_overview(data, evaluations, selected, recommended):
    section('From measurements to model insight',
            'Explore the dataset, estimate potability, and inspect the evidence behind each model.')
    result = evaluations.get(selected, {})
    metric_row([
        ('Test accuracy', metric_value(result, 'test_accuracy'), 'Share of held-out test records classified correctly.'),
        ('Test Macro F1', metric_value(result, 'test_macro_f1'), 'Balances precision and recall, giving both classes equal weight.'),
        ('Selected model', selected.replace('_', ' '), 'The model currently used for predictions and displayed metrics.'),
        ('Dataset records', f'{len(data):,}' if data is not None else 'Unavailable', 'Rows in the original dataset; not live measurements.'),
    ])
    left, right = st.columns([1.2, 1], gap='large')
    with left:
        with st.container(border=True):
            st.markdown('### A transparent prediction workflow')
            st.markdown('**01 / Measure**  \nEnter nine water-quality measurements or upload a CSV.')
            st.markdown('**02 / Estimate**  \nA fitted Pipeline handles missing values, scaling, and classification.')
            st.markdown('**03 / Evaluate**  \nRead class probabilities alongside test metrics and known limitations.')
            st.caption('Dataset-based analysis only. No sensor connection or live monitoring.')
        st.markdown('#### What the evaluation tells us')
        if recommended:
            st.write(f'**{recommended.replace("_", " ")}** is the reported selection by highest cross-validation Macro F1, not test accuracy.')
        else:
            st.info('Model recommendation is unavailable. Run training to generate evaluation results.')
        overfit_notice(result)
    with right:
        with st.container(border=True):
            if data is not None:
                chart(distribution_chart(data), 'overview_classes', 350)
                st.caption('0 = Not Potable · 1 = Potable. Recorded labels are not water-safety certifications.')
            else:
                st.info('Dataset distribution is unavailable.')
        st.caption('Metric source: models/evaluation_results.json. Select a different model in the sidebar to inspect its results.')


def initialize_inputs():
    if 'measurement_values' not in st.session_state:
        st.session_state.measurement_values = {name: spec[3] for name, spec in FIELD_SPECS.items()}
    for name in FEATURES:
        key = f'measure_{name}'
        if key not in st.session_state:
            st.session_state[key] = st.session_state.measurement_values[name]


def load_example(data, pipeline, target=None):
    if data is None or pipeline is None:
        st.warning('A dataset and loaded model are needed to load an example.')
        return
    candidates = data.dropna(subset=FEATURES).copy()
    for name in FEATURES:
        candidates = candidates[candidates[name].between(0, FIELD_SPECS[name][1])]
    if target is not None and not candidates.empty:
        predictions, _, _ = predict_frame(pipeline, candidates)
        candidates = candidates.loc[predictions == target]
    if candidates.empty:
        st.warning('No complete dataset example fits the form ranges and requested model class.')
        return
    row = candidates.sample(1).iloc[0]
    for name in FEATURES:
        st.session_state[f'measure_{name}'] = float(row[name])
    st.session_state.measurement_values = {name: float(row[name]) for name in FEATURES}
    st.session_state.ph_missing = False
    st.session_state.pop('manual_result', None)
    st.caption(f'Loaded dataset row {row.name}. Historical label: {CLASS_LABELS[int(row.Potability)]}; model predictions can differ from recorded labels.')


def render_prediction(data, pipeline, selected, model_version):
    section('Water Prediction', 'Enter one sample. The model estimates a dataset class; it does not determine whether water is safe to drink.')
    initialize_inputs()
    with st.container(horizontal=True, gap='small'):
        if st.button('Load Potable example', disabled=pipeline is None):
            load_example(data, pipeline, 1)
        if st.button('Load Not Potable example', disabled=pipeline is None):
            load_example(data, pipeline, 0)
        if st.button('Load random example', disabled=pipeline is None):
            load_example(data, pipeline)
    ph_missing = st.checkbox('pH measurement unavailable', key='ph_missing',
                             help='The fitted Pipeline will use its training-set median for missing pH.')
    with st.form('water_measurements', border=True):
        groups = [
            ('Physical profile', ['ph', 'Hardness', 'Turbidity']),
            ('Dissolved constituents', ['Solids', 'Sulfate', 'Conductivity']),
            ('Chemical profile', ['Chloramines', 'Organic_carbon', 'Trihalomethanes']),
        ]
        values = {}
        for column, (title, fields) in zip(st.columns(3, gap='large'), groups):
            with column:
                st.markdown(f'#### {title}')
                for name in fields:
                    label, maximum, step, _, help_text = FIELD_SPECS[name]
                    values[name] = st.number_input(label, min_value=0.0, max_value=maximum,
                                                  step=step, key=f'measure_{name}',
                                                  disabled=name == 'ph' and ph_missing, help=help_text)
        st.caption('Blank pH and pH = 0 use the same missing-value handling. Form limits are input limits, not drinking-water standards.')
        submitted = st.form_submit_button('Estimate potability', type='primary',
                                         icon=':material/analytics:', disabled=pipeline is None)
    if submitted:
        st.session_state.measurement_values = values.copy()
        sample = pd.DataFrame([values], columns=FEATURES)
        if ph_missing:
            sample['ph'] = np.nan
        try:
            predictions, probabilities, prepared = predict_frame(pipeline, sample)
            st.session_state.manual_result = dict(
                sample=sample, prepared=prepared, prediction=int(predictions[0]),
                probabilities=probabilities[0] if probabilities is not None else None,
                model=selected, version=model_version,
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state.pop('manual_result', None)
            st.error(f'Prediction could not be completed: {exc}')
    result = st.session_state.get('manual_result')
    if not result:
        st.info('Your result and downloadable report will appear after you submit a sample.', icon=':material/info:')
        return
    if pipeline is None or result['model'] != selected or result['version'] != model_version:
        st.info('The model has changed. Submit the form again to generate a current prediction.')
        return
    st.markdown('#### Latest submitted sample')
    left, right = st.columns([1, 1], gap='large')
    prediction, probabilities = result['prediction'], result['probabilities']
    with left:
        st.html(f'<div class="wqi-result"><div class="wqi-kicker">Predicted class / {prediction}</div>'
                f'<h3>{CLASS_NAMES[prediction]}</h3><p>The model assigns this sample to the '
                f'<strong>{CLASS_NAMES[prediction]}</strong> dataset class. This is not a laboratory finding.</p></div>')
        if probabilities is not None:
            st.metric('Predicted-class probability', f'{probabilities[prediction]:.1%}')
        st.caption(f'Model: {selected.replace("_", " ")}. Probabilities are model outputs, not guaranteed real-world confidence.')
        if result['prepared'].isna().any(axis=None):
            st.info('Missing measurements (including pH = 0) were passed to the Pipeline’s fitted median imputer.')
        st.warning(DISCLAIMER)
        with st.expander('Project reference checks · legacy rules'):
            st.caption('These numerical rules are retained from the original app. They are not a verified standards-compliance assessment and do not override laboratory testing or certify potability.')
            flags = reference_checks(result['sample'])
            for flag in flags:
                st.warning(flag)
            if not flags:
                st.info('No available measurement triggered the legacy reference rules. This does not establish drinking-water safety.')
        report = prediction_report(result['sample'], prediction, probabilities, selected)
        st.download_button('Download prediction report', report.encode('utf-8'),
                           'water_quality_report.txt', mime='text/plain', icon=':material/download:')
    with right:
        if probabilities is not None:
            fig = px.bar(x=probabilities * 100, y=list(CLASS_LABELS.values()), orientation='h',
                         color=list(CLASS_LABELS.values()), color_discrete_map=COLORS,
                         labels={'x': 'Model probability (%)', 'y': 'Target class'},
                         title='Probability assigned to each class', text=[f'{p:.1%}' for p in probabilities])
            fig.update_layout(showlegend=False, xaxis_range=[0, 100])
            chart(fig, 'manual_probabilities', 250)
        else:
            st.info('This model does not expose class probabilities.')
        st.dataframe(result['sample'].T.rename(columns={0: 'Submitted measurement'}), width='stretch')
        st.caption('Measurements retain their original units; unlike quantities are not compared on a shared value axis.')


def render_batch(pipeline, selected):
    section('Batch Analysis', 'Upload measurements, validate the schema, and export one prediction per row.')
    with st.container(border=True):
        st.markdown('#### Prepare your file')
        st.write('Use a comma-separated UTF-8 CSV with the following exact, case-sensitive column names. Extra columns are retained; Potability is optional and is never used as an input.')
        st.code(','.join(FEATURES), language=None, wrap_lines=True)
        st.caption('Use finite numeric values. Leave unknown measurements blank. Zero pH is treated as missing in memory, not rewritten in your original file. Probability output columns use percentages (0–100).')
        template = pd.DataFrame([{name: spec[3] for name, spec in FIELD_SPECS.items()}], columns=FEATURES)
        st.download_button('Download example CSV', template.to_csv(index=False).encode('utf-8'),
                           'water_quality_example.csv', mime='text/csv', icon=':material/download:')
        uploaded = st.file_uploader('Upload measurement CSV', type=['csv'], key='batch_upload')
    if uploaded is None:
        st.info('Choose a CSV to preview and validate its contents. Analysis runs only when requested.')
        return
    content = uploaded.getvalue()
    try:
        frame = read_batch_csv(content)
    except ValueError as exc:
        st.error(f'CSV validation: {exc}')
        return
    signature = (hashlib.sha256(content).hexdigest(), selected,
                 artifact_version(BASE_DIR / 'models' / f'{selected}.pkl'),
                 artifact_version(BASE_DIR / 'models' / 'feature_names.pkl'))
    st.success(f'Validated {len(frame):,} rows and all nine required feature columns.')
    with st.expander('Preview uploaded measurements', expanded=False):
        st.dataframe(frame.head(20), width='stretch', hide_index=True)
    if st.button('Analyze batch', type='primary', icon=':material/analytics:', disabled=pipeline is None):
        try:
            with st.spinner('Running the fitted Pipeline…'):
                results, prepared = batch_results(pipeline, frame, selected)
            st.session_state.batch_result = (signature, results, int(prepared.isna().sum().sum()))
        except (ValueError, TypeError, RuntimeError) as exc:
            st.session_state.pop('batch_result', None)
            st.error(f'Batch prediction failed: {exc}')
    saved = st.session_state.get('batch_result')
    if pipeline is None or saved is None or saved[0] != signature:
        return
    _, results, missing_count = saved
    metric_row([
        ('Samples analyzed', f'{len(results):,}', 'One prediction for each uploaded row.'),
        ('0 = Not Potable', f'{results.Predicted_Class.eq(0).sum():,}', 'Predicted dataset class 0.'),
        ('1 = Potable', f'{results.Predicted_Class.eq(1).sum():,}', 'Predicted dataset class 1.'),
        ('Missing input cells', f'{missing_count:,}', 'Includes zero pH values; imputed only inside the fitted Pipeline.'),
    ])
    st.dataframe(results, width='stretch', hide_index=True)
    st.caption('Original uploaded values are retained in the result table; pH normalization happens only in the prediction copy. Scroll within the table to see all input and output columns.')
    st.download_button('Download batch results', results.to_csv(index=False).encode('utf-8'),
                       'batch_predictions.csv', mime='text/csv', type='primary', icon=':material/download:')
    st.warning(DISCLAIMER)
    if 'Max_Class_Probability' in results:
        chart(px.histogram(results, x='Max_Class_Probability', nbins=20,
                           title='Distribution of highest class probabilities',
                           labels={'Max_Class_Probability': 'Highest model probability (%)'},
                           color_discrete_sequence=['#08949C']), 'batch_probability')


def render_dataset(data):
    section('Dataset Explorer', 'Inspect the original CSV, including recorded missing values and target labels. No dataset values are changed.')
    if data is None:
        st.error('Dataset unavailable. Restore dataset/water_potability.csv to use this view.')
        return
    metric_row([
        ('Dataset records', f'{len(data):,}', 'Original dataset rows.'),
        ('Input features', len(FEATURES), 'The Potability target is not an input feature.'),
        ('Missing feature cells', f'{data[FEATURES].isna().sum().sum():,}', 'Raw CSV missing values; excludes zero pH.'),
        ('Zero pH records', int(data.ph.eq(0).sum()), 'Treated as missing only during preprocessing and prediction.'),
    ])
    left, right = st.columns(2, gap='large')
    with left:
        with st.container(border=True):
            chart(distribution_chart(data), 'dataset_classes')
    with right:
        missing = pd.DataFrame({'Feature': FEATURES, 'Missing cells': data[FEATURES].isna().sum().values})
        with st.container(border=True):
            chart(px.bar(missing.sort_values('Missing cells'), x='Missing cells', y='Feature', orientation='h',
                         title='Missing values in the original CSV', color_discrete_sequence=['#6C8797']), 'dataset_missing')
    feature = st.selectbox('Explore a feature', FEATURES, format_func=lambda name: FIELD_SPECS[name][0])
    plot_data = data[[feature, 'Potability']].copy()
    plot_data['Target class'] = plot_data.Potability.map(CLASS_LABELS)
    fig = px.histogram(plot_data, x=feature, color='Target class', nbins=35, barmode='overlay', opacity=.75,
                       category_orders={'Target class': list(CLASS_LABELS.values())},
                       color_discrete_map=COLORS, title=f'{FIELD_SPECS[feature][0]} by recorded target class',
                       labels={feature: FIELD_SPECS[feature][0]})
    fig.update_yaxes(title='Number of records')
    chart(fig, 'dataset_distribution')
    st.caption('Raw values are plotted without imputation. Missing values are omitted; zero pH remains visible here.')
    with st.expander('Feature correlations'):
        corr = data[FEATURES].corr()
        fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale=['#29485D', '#F7FAFB', '#08949C'],
                        title='Pairwise Pearson correlation · original features', aspect='auto',
                        labels={'x': 'Feature', 'y': 'Feature', 'color': 'Correlation'})
        chart(fig, 'dataset_correlations', 520)
        st.caption('Pairwise complete observations; correlation is not evidence of causation or potability.')
    with st.expander('Preview and descriptive statistics'):
        st.dataframe(data.head(50), width='stretch', hide_index=True)
        st.dataframe(data.describe().T, width='stretch')
        missing_table = data.isna().sum().rename('Missing cells').to_frame()
        missing_table['Missing (%)'] = missing_table['Missing cells'] / len(data) * 100
        st.dataframe(missing_table, width='stretch')


def render_performance(pipeline, evaluations, selected):
    section('Model Performance', 'Saved cross-validation and held-out test results. Accuracy alone does not establish drinking-water safety.')
    result = evaluations.get(selected)
    if not result:
        st.info('No evaluation report is available for this model. Run python train_model.py to generate it.')
        return
    metric_row([
        ('Test accuracy', metric_value(result, 'test_accuracy'), 'Correct classifications on the untouched test split.'),
        ('Test Macro F1', metric_value(result, 'test_macro_f1'), 'F1 averaged equally over both target classes.'),
        ('Test weighted F1', metric_value(result, 'test_weighted_f1'), 'F1 averaged by class frequency.'),
        ('Validation Macro F1', metric_value(result, 'cv_macro_f1_mean'), 'Five-fold mean used for model selection.'),
    ])
    overfit_notice(result)
    comparison = pd.DataFrame([
        {'Model': name.replace('_', ' '), 'Validation Macro F1': r.get('cv_macro_f1_mean'),
         'Test Macro F1': r.get('test_macro_f1'), 'Test accuracy': r.get('test_accuracy'),
         'Test weighted F1': r.get('test_weighted_f1'),
         'CV Macro F1 std': r.get('cv_macro_f1_std')}
        for name, r in evaluations.items()
    ]).sort_values('Validation Macro F1', ascending=False)
    fig = px.bar(comparison.melt(id_vars='Model', value_vars=['Validation Macro F1', 'Test Macro F1'],
                                 var_name='Evaluation', value_name='Score'),
                 x='Score', y='Model', color='Evaluation', orientation='h', barmode='group',
                 title='Model comparison · Macro F1', color_discrete_sequence=['#29485D', '#08949C'])
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange='reversed')
    chart(fig, 'performance_comparison', 400)
    st.dataframe(comparison, width='stretch', hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format='%.4f')
                                for name in comparison.columns if name != 'Model'})
    st.caption('Baseline Majority always predicts the most frequent training class. Selection uses validation Macro F1, never test scores.')
    left, right = st.columns(2, gap='large')
    with left:
        cm = result.get('confusion_matrix')
        if cm is not None and np.asarray(cm).shape == (2, 2):
            fig = go.Figure(go.Heatmap(z=cm, x=list(CLASS_LABELS.values()), y=list(CLASS_LABELS.values()),
                                      text=cm, texttemplate='%{text}', showscale=False,
                                      colorscale=['#EDF3F5', '#087E8B'],
                                      hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Records: %{z}<extra></extra>'))
            fig.update_layout(title=f'Confusion matrix · {selected.replace("_", " ")}',
                              xaxis_title='Predicted class (columns)', yaxis_title='Actual class (rows)')
            fig.update_yaxes(autorange='reversed')
            chart(fig, 'performance_confusion', 370)
        else:
            st.info('Confusion matrix unavailable.')
        st.caption('Rows = actual classes. Columns = predicted classes. 0 = Not Potable · 1 = Potable.')
    with right:
        st.markdown('#### Per-class test performance')
        per_class = pd.DataFrame([
            {'Target class': CLASS_LABELS[label],
             'Precision': result.get(f'test_precision_class_{label}'),
             'Recall': result.get(f'test_recall_class_{label}'),
             'F1': result.get(f'test_f1_class_{label}')}
            for label in (0, 1)
        ])
        st.dataframe(per_class, width='stretch', hide_index=True,
                     column_config={name: st.column_config.NumberColumn(format='%.3f') for name in ['Precision', 'Recall', 'F1']})
        st.caption('Precision: agreement among predictions for a class. Recall: fraction of actual class records recovered. F1: balance of precision and recall.')
        if cm is not None and np.asarray(cm).shape == (2, 2):
            st.warning(f'{cm[0][1]:,} test records labeled 0 = Not Potable were predicted as 1 = Potable. This is a concrete limitation, not a safety guarantee.')
    stages = pd.DataFrame({'Evaluation stage': ['Fold training (mean)', 'Fold validation (mean)', 'Held-out test'],
                           'Accuracy': [result.get('train_accuracy_mean'), result.get('cv_accuracy_mean'), result.get('test_accuracy')],
                           'Standard deviation': [result.get('train_accuracy_std', 0), result.get('cv_accuracy_std', 0), 0]})
    fig = px.bar(stages, x='Evaluation stage', y='Accuracy', error_y='Standard deviation',
                 title='Training versus unseen data · accuracy', color_discrete_sequence=['#29485D'])
    fig.update_yaxes(range=[0, 1.05])
    chart(fig, 'performance_overfit', 330)
    st.caption('Error bars show standard deviation across CV folds, not confidence intervals. Test accuracy is a single held-out evaluation.')
    if pipeline is not None:
        importance, label = feature_importance(pipeline)
        if importance is not None:
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                         title=f'{label} · {selected.replace("_", " ")}', color_discrete_sequence=['#08949C'])
            fig.update_xaxes(title=label)
            chart(fig, 'performance_importance', 370)
            st.caption('Global model associations, not a causal explanation or scientific threshold. Tree importance can be biased; coefficient magnitudes are not tree importances.')
    with st.container(horizontal=True):
        for filename, label, mime in [('evaluation_report.txt', 'Download evaluation report', 'text/plain'),
                                      ('evaluation_results.json', 'Download evaluation JSON', 'application/json')]:
            path = BASE_DIR / 'models' / filename
            if path.exists():
                st.download_button(label, path.read_bytes(), filename, mime=mime, icon=':material/download:')


def render_methodology(evaluations, selected):
    section('Training & Methodology', 'What the training script actually does, and where this educational model falls short.')
    with st.container(border=True):
        st.markdown('#### Reproducible, Pipeline-based classification')
        st.markdown('''
**01 / Source and target**

`dataset/water_potability.csv` supplies nine numeric features and the `Potability` target. **0 = Not Potable** and **1 = Potable** are recorded dataset classes. The original CSV is not rewritten.

**02 / Prepare and split**

The shared `treat_zero_ph_as_missing` helper copies inputs and converts `ph == 0` to `NaN`. The project treats zero pH as a missing-data marker; this is a dataset-handling convention, not a claim that pH zero is physically impossible. A stratified **80% training / 20% test** split uses `random_state=42`.

**03 / Fit preprocessing inside every fold**

Each saved Pipeline contains `SimpleImputer(strategy="median") → StandardScaler → classifier`. Five-fold `StratifiedKFold(shuffle=True, random_state=42)` cross-validation fits preprocessing separately in each training fold. The UI never imputes or scales inputs a second time.

**04 / Compare and select**

Random Forest (200 trees, depth 10), XGBoost (200 estimators, depth 5), Decision Tree (depth 10), and Logistic Regression (1,000 maximum iterations) are compared with a majority-class baseline. The highest **mean validation Macro F1** selects the best model. Random Forest, Decision Tree, and Logistic Regression use balanced class weights; XGBoost uses `scale_pos_weight`.

**05 / Final fit and evaluation**

Every candidate is refitted on the full training partition and evaluated on the held-out test partition. Four complete model Pipelines, the feature schema, evaluation JSON, and a readable report are saved under `models/`. The baseline is evaluated but not saved as a prediction artifact.
''')
    st.markdown('#### Observed limitations — not production validation')
    overfit_notice(evaluations.get(selected, {}))
    st.markdown('''
- **Imperfect generalization.** Inspect both class recalls and the confusion matrix; overall accuracy can hide class-specific errors.
- **Dataset quality and scope.** Missing measurements and class imbalance affect learning. The dataset does not establish coverage of new locations, instruments, contaminants, or time periods.
- **Imbalance-weighting caveat.** The training script computes XGBoost’s class ratio from the full dataset before splitting. Preprocessing is fold-local, but this ratio uses held-out label counts. The existing methodology is preserved and should be reconsidered in future validation.
- **Probabilities are not guarantees.** These classifier outputs have not been demonstrated to be calibrated probabilities of real-world drinking-water safety.
- **Importance is not causation.** Feature rankings describe the fitted model, not approved scientific limits or treatment recommendations.
- **No external validation.** A single dataset split and cross-validation do not establish scientific, medical, or production readiness. Laboratory testing remains essential.
''')
    st.warning(DISCLAIMER)
    with st.expander('Train and verify locally'):
        st.code('python train_model.py\npython test_models.py\npython test_app.py\npython -m streamlit run app.py', language='bash')
        st.caption('Training intentionally replaces the four fitted Pipeline artifacts and evaluation reports. It does not modify the original dataset. Legacy standalone scaler/imputer files are not used by the app.')


def main():
    st.html(SHELL_CSS)
    evaluations, data, errors = {}, None, []
    try:
        evaluations = cached_evaluation(artifact_version(BASE_DIR / 'models' / 'evaluation_results.json'))
    except (OSError, ValueError) as exc:
        errors.append(f'Evaluation report unavailable: {exc}')
    try:
        data = cached_dataset(artifact_version(BASE_DIR / 'dataset' / 'water_potability.csv'))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        errors.append(f'Dataset unavailable: {exc}')
    candidates = {name: r['cv_macro_f1_mean'] for name, r in evaluations.items()
                  if name in MODEL_NAMES and isinstance(r.get('cv_macro_f1_mean'), (int, float))
                  and np.isfinite(r['cv_macro_f1_mean'])}
    recommended = max(candidates, key=lambda name: candidates[name]) if candidates else None
    with st.sidebar:
        st.markdown(':material/water_drop: **WQI / ANALYTICS**')
        st.caption('WATER QUALITY INTELLIGENCE')
        st.divider()
        view = st.radio('Workspace', NAVIGATION, key='workspace')
        st.divider()
        selected = st.selectbox('Prediction model', MODEL_NAMES,
                                index=MODEL_NAMES.index(recommended) if recommended else 0,
                                format_func=lambda name: name.replace('_', ' '), key='selected_model')
        if recommended:
            st.caption(f'CV-selected model: {recommended.replace("_", " ")}')
        st.caption('The selected model powers both manual and batch predictions.')
    models, model_errors = {}, {}
    schema_version = artifact_version(BASE_DIR / 'models' / 'feature_names.pkl')
    for name in MODEL_NAMES:
        try:
            models[name] = cached_pipeline(name, artifact_version(BASE_DIR / 'models' / f'{name}.pkl'), schema_version)
        except Exception as exc:
            # Artifact deserialization can raise backend-specific exceptions; keep other views accessible.
            model_errors[name] = str(exc)
    pipeline = models.get(selected)
    status = (f'{selected.replace("_", " ")} loaded · {len(models)}/4 Pipelines available'
              if pipeline is not None else f'{selected.replace("_", " ")} unavailable · {len(models)}/4 Pipelines available')
    st.html('<header class="wqi-hero"><div class="wqi-eyebrow">ENVIRONMENTAL DATA / MACHINE LEARNING</div>'
            '<h1>Water Quality Intelligence</h1>'
            '<p>Understand water-quality measurements through transparent potability predictions, '
            'dataset exploration, and model evaluation.</p>'
            f'<div class="wqi-status">{escape(status)}</div></header>')
    st.info(DISCLAIMER, icon=':material/info:')
    for error in errors:
        st.warning(error)
    if model_errors:
        with st.expander('Model loading details', expanded=pipeline is None):
            for name, error in model_errors.items():
                st.error(f'{name}: {error}')
            st.code('python train_model.py', language='bash')
    version = (artifact_version(BASE_DIR / 'models' / f'{selected}.pkl'), schema_version)
    if view == 'Overview':
        render_overview(data, evaluations, selected, recommended)
    elif view == 'Water Prediction':
        render_prediction(data, pipeline, selected, version)
    elif view == 'Batch Analysis':
        render_batch(pipeline, selected)
    elif view == 'Dataset Explorer':
        render_dataset(data)
    elif view == 'Model Performance':
        render_performance(pipeline, evaluations, selected)
    else:
        render_methodology(evaluations, selected)
    st.html('<footer class="wqi-footer"><strong>Water Quality Intelligence</strong> · Dataset-based potability classification'
            '<br>Python / Pandas / NumPy / scikit-learn Pipelines / XGBoost / Streamlit / Plotly'
            '<br>Educational use only. Not a drinking-water safety certification.</footer>')


if __name__ == '__main__':
    main()
