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
    MODEL_NAMES, batch_results, decision_threshold, feature_importance, load_pipeline,
    predict_frame, prediction_report, read_batch_csv, reference_checks,
)

NAVIGATION = [
    {'label': 'Overview', 'icon': ':material/dashboard:', 'group': 'WORKSPACE'},
    {'label': 'Water Prediction', 'icon': ':material/science:', 'group': 'WORKSPACE'},
    {'label': 'Batch Analysis', 'icon': ':material/upload_file:', 'group': 'WORKSPACE'},
    {'label': 'Dataset Explorer', 'icon': ':material/analytics:', 'group': 'ANALYTICS'},
    {'label': 'Model Performance', 'icon': ':material/leaderboard:', 'group': 'ANALYTICS'},
    {'label': 'Training & Methodology', 'icon': ':material/psychology:', 'group': 'PROJECT'},
]
NAV_ICONS = {item['label']: item['icon'] for item in NAVIGATION}

COLORS_LIGHT = {CLASS_LABELS[0]: '#29485D', CLASS_LABELS[1]: '#08949C'}
COLORS_DARK = {CLASS_LABELS[0]: '#475569', CLASS_LABELS[1]: '#08949C'}

st.set_page_config(
    page_title='Water Quality Intelligence',
    page_icon=':material/water_drop:',
    layout='wide',
    initial_sidebar_state='expanded',
)


def get_css(dark_mode: bool = False) -> str:
    if dark_mode:
        navy = '#F8FAFC'
        navy_light = '#1E293B'
        slate_bg = '#0B131E'
        card_bg = '#162032'
        border = '#26354A'
        border_strong = '#3B4D66'
        text_main = '#F1F5F9'
        text_secondary = '#94A3B8'
        text_muted = '#64748B'
        sidebar_bg = '#0A131F'
        sidebar_hover = '#162436'
        sidebar_text = '#CBD5E1'
        sidebar_heading = '#7EB2CE'
        topbar_bg = '#162032'
        metric_bg = '#162032'
        metric_val = '#F8FAFC'
    else:
        navy = '#0F2537'
        navy_light = '#16364D'
        slate_bg = '#F8FAFC'
        card_bg = '#FFFFFF'
        border = '#E2E8F0'
        border_strong = '#CBD5E1'
        text_main = '#0F172A'
        text_secondary = '#475569'
        text_muted = '#64748B'
        sidebar_bg = '#0F2537'
        sidebar_hover = '#19384E'
        sidebar_text = '#C5D7E3'
        sidebar_heading = '#7EB2CE'
        topbar_bg = '#FFFFFF'
        metric_bg = '#FFFFFF'
        metric_val = '#0F2537'

    return f"""
<style>
:root {{
  --navy: {navy};
  --navy-light: {navy_light};
  --teal: #087E8B;
  --teal-hover: #06646E;
  --teal-active: #0A939E;
  --slate-bg: {slate_bg};
  --card-bg: {card_bg};
  --border: {border};
  --border-strong: {border_strong};
  --text-main: {text_main};
  --text-secondary: {text_secondary};
  --text-muted: {text_muted};
  --sidebar-bg: {sidebar_bg};
  --sidebar-hover: {sidebar_hover};
  --sidebar-text: {sidebar_text};
  --sidebar-heading: {sidebar_heading};
}}

/* Base Layout & Clean Minimal Background */
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--slate-bg) !important;
  color: var(--text-main) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* Streamlit Header Fix: Ensure header does not obscure top content */
header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 2.75rem !important;
  z-index: 99 !important;
}}

/* Generous top padding so the top app bar is 100% visible and never clipped */
.block-container {{
  padding-top: 4.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1240px !important;
}}

/* Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {{
  color: var(--navy) !important;
  letter-spacing: -0.3px;
  font-weight: 700;
}}
p, span, label {{
  color: var(--text-main);
}}

/* High-Contrast Navigation Sidebar */
[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid #1D3B50 !important;
}}
[data-testid="stSidebar"] * {{
  color: var(--sidebar-text);
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
  color: var(--sidebar-text) !important;
}}
[data-testid="stSidebarCollapseButton"] button {{
  color: #FFFFFF !important;
}}
[data-testid="stSidebar"] hr {{
  border-color: rgba(126, 178, 206, 0.2) !important;
  margin: 1rem 0 !important;
}}

/* Sidebar Branding Header */
.sidebar-brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 0 14px 0;
  border-bottom: 1px solid rgba(126, 178, 206, 0.2);
  margin-bottom: 10px;
}}
.sidebar-brand-icon {{
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: linear-gradient(135deg, #08949C 0%, #065F66 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 1.1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.25);
}}
.sidebar-brand-title {{
  font-size: 0.95rem;
  font-weight: 700;
  color: #FFFFFF !important;
  letter-spacing: -0.2px;
}}
.sidebar-brand-subtitle {{
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--sidebar-heading) !important;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}}

/* Sidebar Navigation Items - Clean & Legible */
[data-testid="stSidebar"] [data-testid="stRadio"] {{
  margin-top: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{
  display: flex !important;
  flex-direction: column !important;
  gap: 2px !important;
}}

/* Style each navigation item */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {{
  background: transparent !important;
  padding: 7px 12px !important;
  border-radius: 6px !important;
  margin: 1px 0 !important;
  cursor: pointer !important;
  transition: all 0.15s ease-in-out !important;
  display: flex !important;
  align-items: center !important;
  border-left: 3px solid transparent !important;
}}

/* Unselected hover */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {{
  background: var(--sidebar-hover) !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {{
  color: var(--sidebar-text) !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  margin: 0 !important;
  line-height: 1.3 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover p {{
  color: #FFFFFF !important;
}}

/* Active Navigation Item: Crisp teal pill with high contrast */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
  background: rgba(8, 148, 156, 0.22) !important;
  border-left: 3px solid #38E1E6 !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{
  color: #FFFFFF !important;
  font-weight: 600 !important;
}}

/* Group Headers Positioned Above Item 1, 4, 6 */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(1) {{
  margin-top: 26px !important;
  position: relative !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(1)::before {{
  content: "WORKSPACE";
  position: absolute;
  top: -22px;
  left: 6px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: var(--sidebar-heading);
  text-transform: uppercase;
  pointer-events: none;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(4) {{
  margin-top: 34px !important;
  position: relative !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(4)::before {{
  content: "ANALYTICS";
  position: absolute;
  top: -22px;
  left: 6px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: var(--sidebar-heading);
  text-transform: uppercase;
  pointer-events: none;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(4)::after {{
  content: "";
  position: absolute;
  top: -29px;
  left: 6px;
  right: 6px;
  height: 1px;
  background: rgba(126, 178, 206, 0.2);
}}

[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(6) {{
  margin-top: 34px !important;
  position: relative !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(6)::before {{
  content: "PROJECT";
  position: absolute;
  top: -22px;
  left: 6px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: var(--sidebar-heading);
  text-transform: uppercase;
  pointer-events: none;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(6)::after {{
  content: "";
  position: absolute;
  top: -29px;
  left: 6px;
  right: 6px;
  height: 1px;
  background: rgba(126, 178, 206, 0.2);
}}

/* Model & Theme Section in Sidebar */
.sidebar-model-section {{
  margin-top: 1.25rem;
  padding-top: 0.85rem;
  border-top: 1px solid rgba(126, 178, 206, 0.2);
}}
.sidebar-section-label {{
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 1.2px;
  color: var(--sidebar-heading);
  text-transform: uppercase;
  margin-bottom: 8px;
}}
[data-testid="stSidebar"] [data-baseweb="select"] {{
  background-color: var(--navy-light) !important;
  border: 1px solid #285473 !important;
  border-radius: 6px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: #FFFFFF !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
  fill: #8EB0C4 !important;
}}
.model-badge {{
  font-size: 0.72rem;
  color: #56CFD0;
  margin-top: 6px;
  font-weight: 600;
}}

/* Top App Bar */
.app-topbar {{
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  flex-wrap: wrap !important;
  gap: 10px !important;
  padding: 8px 16px !important;
  background: {topbar_bg} !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}}
.app-topbar-left {{
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  flex-wrap: wrap !important;
}}
.app-topbar-brand {{
  font-size: 0.84rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  color: var(--navy) !important;
  white-space: nowrap !important;
}}
.app-topbar-divider {{
  color: var(--border-strong);
  font-size: 0.85rem;
}}
.app-topbar-subtitle {{
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
  font-weight: 500 !important;
  white-space: nowrap !important;
}}
.status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.76rem;
  padding: 4px 10px;
  border-radius: 20px;
  font-weight: 500;
  white-space: nowrap !important;
}}
.status-pill-success {{
  background: {'#133748' if dark_mode else '#E6F7F7'} !important;
  color: {'#38E1E6' if dark_mode else '#06646E'} !important;
  border: 1px solid {'#1D5A6E' if dark_mode else '#B8EBEF'} !important;
}}
.status-pill-warning {{
  background: #FEF3C7 !important;
  color: #92400E !important;
  border: 1px solid #FCD34D !important;
}}
.status-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--teal);
}}

/* Compact Page Header */
.page-header-container {{
  margin-bottom: 1.25rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border);
}}
.page-header-title-row {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.page-title {{
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--navy) !important;
  letter-spacing: -0.4px;
  margin: 0;
  line-height: 1.25;
}}
.page-header-badge {{
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  background: {'#162032' if dark_mode else '#EDF3F5'} !important;
  color: {'#7EB2CE' if dark_mode else 'var(--navy)'} !important;
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid var(--border) !important;
}}
.page-description {{
  font-size: 0.92rem;
  color: var(--text-secondary) !important;
  margin: 4px 0 0 0;
  line-height: 1.45;
}}

/* Aligned Metric Cards - NO TEXT CLIPPING */
[data-testid="stMetric"] {{
  background: {metric_bg} !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 10px 14px !important;
  min-height: 96px !important;
  height: auto !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
  overflow: visible !important;
  min-width: 0 !important;
}}
[data-testid="stMetricLabel"] {{
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
  color: var(--text-muted) !important;
  white-space: normal !important;
  word-break: break-word !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.25 !important;
}}
[data-testid="stMetricLabel"] p {{
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}}
[data-testid="stMetricValue"] {{
  font-size: clamp(1.15rem, 1.8vw, 1.5rem) !important;
  font-weight: 700 !important;
  color: {metric_val} !important;
  letter-spacing: -0.5px !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  word-break: break-all !important;
  line-height: 1.2 !important;
}}
[data-testid="stMetricValue"] div {{
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}}

/* Standard Cards and Containers */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 8px;
  border: 1px solid var(--border) !important;
  background: var(--card-bg) !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}}

/* Buttons */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {{
  background: var(--teal) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.2rem !important;
  box-shadow: 0 1px 2px rgba(8, 126, 139, 0.2) !important;
  transition: all 0.15s ease !important;
}}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {{
  background: var(--teal-hover) !important;
  box-shadow: 0 2px 4px rgba(8, 126, 139, 0.3) !important;
}}

/* Clean Tables & Dataframes */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--border) !important;
  border-radius: 6px;
  overflow: hidden;
}}

/* Restrained Alert Styling */
[data-testid="stAlert"] {{
  border-radius: 6px !important;
  font-size: 0.85rem !important;
  padding: 10px 14px !important;
  margin-top: 0.5rem !important;
  margin-bottom: 0.75rem !important;
}}

/* Footer */
.app-footer {{
  border-top: 1px solid var(--border);
  margin-top: 2.5rem;
  padding-top: 1.2rem;
  color: var(--text-muted) !important;
  font-size: 0.8rem;
  line-height: 1.6;
  text-align: center;
}}
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
    cols = st.columns(len(items), gap='small')
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text, border=True)


def page_header(title, description, badge=None):
    badge_html = f'<span class="page-header-badge">{escape(badge)}</span>' if badge else ''
    st.markdown(f'''
    <div class="page-header-container">
      <div class="page-header-title-row">
        <h1 class="page-title">{escape(title)}</h1>
        {badge_html}
      </div>
      <p class="page-description">{escape(description)}</p>
    </div>
    ''', unsafe_allow_html=True)


def chart(figure, key, height=340, dark_mode: bool = False):
    template = 'plotly_dark' if dark_mode else 'plotly_white'
    font_color = '#F1F5F9' if dark_mode else '#1E293B'
    grid_color = '#26354A' if dark_mode else '#E2E8F0'
    title_color = '#F8FAFC' if dark_mode else '#0F2537'

    figure.update_layout(
        template=template, height=height,
        font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif', size=12, color=font_color),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=16, t=46, b=24),
        title_font=dict(size=14, color=title_color), legend_title_text='',
        colorway=['#29485D', '#08949C', '#6C8797', '#C28B45'] if not dark_mode else ['#38E1E6', '#08949C', '#94A3B8', '#F59E0B'],
    )
    figure.update_xaxes(automargin=True, gridcolor=grid_color, zeroline=False)
    figure.update_yaxes(automargin=True, gridcolor=grid_color, zeroline=False)
    st.plotly_chart(figure, width='stretch', key=key, config={'displaylogo': False})


def distribution_chart(data, dark_mode: bool = False):
    counts = data['Potability'].value_counts().reindex([0, 1], fill_value=0)
    color_map = COLORS_DARK if dark_mode else COLORS_LIGHT
    text_color = '#F8FAFC' if dark_mode else '#0F2537'

    fig = go.Figure(go.Pie(
        labels=[CLASS_LABELS[0], CLASS_LABELS[1]], values=counts.values, hole=.72,
        sort=False, marker_colors=list(color_map.values()), textinfo='percent',
        hovertemplate='%{label}<br>%{value:,} records · %{percent}<extra></extra>',
    ))
    fig.update_layout(title='Recorded target classes',
                      legend=dict(orientation='h', y=-.12, x=.5, xanchor='center'),
                      annotations=[dict(text=f'{len(data):,}<br>records', x=.5, y=.5,
                                        font_size=18, showarrow=False, font=dict(color=text_color, family='sans-serif'))])
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


def init_sidebar_nav():
    labels = [item['label'] for item in NAVIGATION]
    return st.radio('Workspace', labels, key='workspace',
                    format_func=lambda name: f"{NAV_ICONS.get(name, '')}  {name}",
                    label_visibility='collapsed')


def render_app_bar(selected, pipeline, recommended, total_models, dark_mode: bool = False):
    status_cls = "status-pill-success" if pipeline is not None else "status-pill-warning"
    status_msg = f"{selected.replace('_', ' ')} active · {total_models}/7 Pipelines ready" if pipeline is not None else f"{selected.replace('_', ' ')} unavailable"
    theme_label = "🌙 Dark Mode" if dark_mode else "☀️ Light Mode"
    st.markdown(f'''
    <div class="app-topbar">
      <div class="app-topbar-left">
        <span class="app-topbar-brand">💧 WATER QUALITY INTELLIGENCE</span>
        <span class="app-topbar-divider">/</span>
        <span class="app-topbar-subtitle">Environmental Analytics</span>
        <span class="app-topbar-divider">/</span>
        <span class="page-header-badge">{escape(theme_label)}</span>
      </div>
      <div>
        <span class="status-pill {status_cls}">
          <span class="status-dot"></span>
          <strong>{escape(status_msg)}</strong>
        </span>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def render_overview(data, evaluations, selected, recommended, dark_mode: bool = False):
    page_header('System Overview', 'Executive summary of dataset attributes, selected pipeline benchmarks, and classification workflow.', 'Analytics Dashboard')
    result = evaluations.get(selected, {})

    # Four aligned KPI cards (Test accuracy first to preserve test assertion when evaluation fails)
    metric_row([
        ('Test accuracy', metric_value(result, 'test_accuracy'), 'Share of held-out test records classified correctly.'),
        ('Test Macro F1', metric_value(result, 'test_macro_f1'), 'Balances precision and recall, giving both classes equal weight.'),
        ('Dataset records', f'{len(data):,}' if data is not None else 'Unavailable', 'Rows in the original dataset; not live measurements.'),
        ('Input features', f'{len(FEATURES)}', 'Nine physical and chemical water quality parameters.'),
    ])

    st.markdown('---')
    left, right = st.columns([1.15, 1], gap='large')

    with left:
        # Selected Model Summary Card
        with st.container(border=True):
            st.markdown('#### Active Pipeline & Diagnostics')
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f'**Active Model**  \n<span style="font-size:1.15rem;font-weight:700;color:var(--navy)">{selected.replace("_", " ")}</span>', unsafe_allow_html=True)
                st.caption('Selected pipeline powering single and batch inference.')
            with col_m2:
                st.markdown('**Selection Criterion**  \nNested CV Macro F1')
                if recommended:
                    st.caption(f'Recommended by CV: **{recommended.replace("_", " ")}**')
                else:
                    st.caption('Model recommendation unavailable.')
            overfit_notice(result)

        # Prediction Workflow Overview Card
        with st.container(border=True):
            st.markdown('#### Three-Stage Analytics Workflow')
            st.markdown('''
            **01 / Measurement Ingestion**  
            Enter 9 laboratory or field parameters manually or batch-upload a CSV table.

            **02 / Pipeline Preprocessing & Scaling**  
            Scikit-learn `Pipeline` applies median imputation (inside folds) and standard z-score scaling.

            **03 / Calibrated Classification & Diagnostics**  
            Probability output evaluated against high-precision threshold rule to reduce false potable risk.
            ''')
            st.caption('Dataset-based statistical inference. No live hardware sensor link.')

    with right:
        # Dataset Class Distribution Card
        with st.container(border=True):
            st.markdown('#### Target Class Distribution')
            if data is not None:
                chart(distribution_chart(data, dark_mode), 'overview_classes', 280, dark_mode)
                st.caption('Historical dataset labels: 0 = Not Potable (61.0%) · 1 = Potable (39.0%). Not laboratory safety certifications.')
            else:
                st.info('Dataset distribution is unavailable.')

        # Missing-Value Summary Card
        with st.container(border=True):
            st.markdown('#### Missing Data Audit (Raw CSV)')
            if data is not None:
                missing_summary = pd.DataFrame({
                    'Parameter': ['Sulfate', 'pH', 'Trihalomethanes', 'Other 6 Parameters'],
                    'Missing Records': [int(data['Sulfate'].isna().sum()), int(data['ph'].isna().sum()), int(data['Trihalomethanes'].isna().sum()), 0],
                    'Missing Rate': [f"{data['Sulfate'].isna().mean():.1%}", f"{data['ph'].isna().mean():.1%}", f"{data['Trihalomethanes'].isna().mean():.1%}", "0.0%"],
                    'Pipeline Strategy': ['Fitted Median', 'Fitted Median (0 treated as NaN)', 'Fitted Median', 'No Imputation Needed']
                })
                st.dataframe(missing_summary, width='stretch', hide_index=True)
                st.caption('Missingness handled internally by fitted Pipeline transformers. Original CSV files remain untouched.')
            else:
                st.info('Missing-value data unavailable.')


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


def render_prediction(data, pipeline, selected, model_version, dark_mode: bool = False):
    page_header('Water Prediction', 'Single-sample potability classification using fitted scikit-learn Pipeline with median imputation and scaling.', 'Inference Engine')
    initialize_inputs()

    # 1. Compact Action Bar for Examples & Options
    with st.container(border=True):
        st.markdown('##### Load Reference Samples & Input Configuration')
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2], gap='small')
        with c1:
            if st.button('Load Potable example', disabled=pipeline is None, use_container_width=True):
                load_example(data, pipeline, 1)
        with c2:
            if st.button('Load Not Potable example', disabled=pipeline is None, use_container_width=True):
                load_example(data, pipeline, 0)
        with c3:
            if st.button('Load random example', disabled=pipeline is None, use_container_width=True):
                load_example(data, pipeline)
        with c4:
            ph_missing = st.checkbox('pH measurement unavailable', key='ph_missing',
                                     help='The fitted Pipeline will use its training-set median for missing pH.')

    # 2. Balanced Three-Column Form Layout
    groups = [
        ('Physical Profile', ['ph', 'Hardness', 'Turbidity']),
        ('Dissolved Constituents', ['Solids', 'Sulfate', 'Conductivity']),
        ('Chemical Profile', ['Chloramines', 'Organic_carbon', 'Trihalomethanes']),
    ]

    with st.form('water_measurements', border=True):
        st.markdown('##### Enter Sample Measurements')
        col1, col2, col3 = st.columns(3, gap='medium')
        cols = [col1, col2, col3]

        for idx, (title, fields) in enumerate(groups):
            with cols[idx]:
                st.markdown(f'<div style="font-size:0.88rem;font-weight:700;color:var(--navy);margin-bottom:8px;padding-bottom:4px;border-bottom:2px solid var(--border)">{escape(title)}</div>', unsafe_allow_html=True)
                for name in fields:
                    label, maximum, step, _, help_text = FIELD_SPECS[name]
                    st.number_input(label, min_value=0.0, max_value=maximum,
                                    step=step, key=f'measure_{name}',
                                    disabled=name == 'ph' and ph_missing, help=help_text)

        st.caption('Blank pH and pH = 0 use the same missing-value handling. Form limits are input limits, not drinking-water standards.')
        submitted = st.form_submit_button('Estimate potability', type='primary', icon=':material/analytics:', disabled=pipeline is None, use_container_width=True)

    if submitted:
        values = {name: st.session_state[f'measure_{name}'] for name in FEATURES}
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
        st.info('Submit a sample to see its predicted class, probability, and model information.')
        return
    if pipeline is None or result['model'] != selected or result['version'] != model_version:
        st.info('The model has changed. Submit the form again to generate a current prediction.')
        return

    # 3. Dedicated Results Area
    st.markdown('---')
    st.markdown('### Prediction Result & Diagnostics')
    prediction, probabilities = result['prediction'], result['probabilities']
    left, right = st.columns([1, 1], gap='large')

    with left:
        is_potable = (prediction == 1)
        if dark_mode:
            badge_bg = "#064E3B" if is_potable else "#1E293B"
            badge_color = "#34D399" if is_potable else "#CBD5E1"
            badge_border = "#059669" if is_potable else "#475569"
        else:
            badge_bg = "#E6F7F7" if is_potable else "#F1F5F9"
            badge_color = "#06646E" if is_potable else "#1E293B"
            badge_border = "#08949C" if is_potable else "#94A3B8"

        st.markdown(f'''
        <div style="background:{badge_bg};border:1px solid {badge_border};border-left:5px solid {badge_border};border-radius:8px;padding:18px 20px;margin-bottom:1rem;">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:{badge_color}">Prediction Classification</div>
          <div style="font-size:1.6rem;font-weight:800;color:{badge_color};margin:4px 0 8px 0;">{CLASS_NAMES[prediction]}</div>
          <div style="font-size:0.85rem;color:var(--text-secondary);">Fitted Model: <strong>{selected.replace("_", " ")}</strong></div>
        </div>
        ''', unsafe_allow_html=True)

        st.caption('This is a model estimate of a dataset class, not a laboratory finding.')
        threshold = decision_threshold(pipeline)
        if threshold is not None:
            st.caption(f'Decision rule: Potable is predicted only if P(Potable) ≥ {threshold:.0%}, a cautious cut-off chosen from training data to reduce false Potable predictions.')
        if probabilities is not None:
            st.markdown(f'**Predicted-class probability:** <span style="font-size:1.15rem;font-weight:700;color:var(--teal)">{probabilities[prediction]:.1%}</span>', unsafe_allow_html=True)
            st.caption('Model probabilities are outputs, not guaranteed real-world confidence.')
        if result['prepared'].isna().any(axis=None):
            st.info('Missing measurements (including pH = 0) were passed to the Pipeline’s fitted median imputer.')
        st.warning(DISCLAIMER)

    with right:
        if probabilities is not None:
            color_map = COLORS_DARK if dark_mode else COLORS_LIGHT
            fig = px.bar(x=probabilities * 100, y=list(CLASS_LABELS.values()), orientation='h',
                         color=list(CLASS_LABELS.values()), color_discrete_map=color_map,
                         labels={'x': 'Model probability (%)', 'y': 'Target class'},
                         title='Probability assigned to each class', text=[f'{p:.1%}' for p in probabilities])
            fig.update_layout(showlegend=False, xaxis_range=[0, 100])
            chart(fig, 'manual_probabilities', 250, dark_mode)
        else:
            st.info('This model does not expose class probabilities.')
        st.dataframe(result['sample'].T.rename(columns={0: 'Submitted measurement'}), width='stretch')
        st.caption('Measurements retain their original units.')
        report = prediction_report(result['sample'], prediction, probabilities, selected, threshold)
        st.download_button('Download prediction report', report.encode('utf-8'),
                           'water_quality_report.txt', mime='text/plain', icon=':material/download:', use_container_width=True)


def render_batch(pipeline, selected, dark_mode: bool = False):
    page_header('Batch Analysis', 'High-throughput CSV ingestion, automated schema validation, and pipeline-based batch scoring.', 'Batch Processor')

    # Step 1: File Requirements & Template Download (Separated from upload control!)
    with st.container(border=True):
        st.markdown('##### 1 · File Format Requirements & Template')
        st.markdown('Upload a comma-separated UTF-8 CSV containing the nine required water quality parameters. Extra columns (e.g. sample IDs, site codes) are automatically preserved in the export. `Potability` is ignored if present.')
        st.code(','.join(FEATURES), language=None, wrap_lines=True)
        st.caption('Use finite numeric values. Blank cells and pH = 0 are treated as missing and imputed by the fitted Pipeline. Output probabilities are reported as percentages (0–100).')
        template = pd.DataFrame([{name: spec[3] for name, spec in FIELD_SPECS.items()}], columns=FEATURES)
        st.download_button('Download example CSV', template.to_csv(index=False).encode('utf-8'),
                           'water_quality_example.csv', mime='text/csv', icon=':material/download:')

    # Step 2: Upload CSV & Automated Validation
    with st.container(border=True):
        st.markdown('##### 2 · Upload & Validation')
        uploaded = st.file_uploader('Upload measurement CSV', type=['csv'], key='batch_upload')

        if uploaded is None:
            st.info('Select a CSV file to validate schema and preview rows. Prediction pipeline runs on command.')
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
        st.success(f'✓ Schema Verified: {len(frame):,} rows and all 9 required feature columns successfully validated.')
        st.caption(f'Source File: {getattr(uploaded, "name", "uploaded.csv")} · Size: {getattr(uploaded, "size", len(content)):,} bytes')

    # Step 3: Data Preview
    with st.expander('3 · Preview Uploaded Measurements', expanded=True):
        st.dataframe(frame.head(20), width='stretch', hide_index=True)
        st.caption(f'Showing initial {min(20, len(frame))} of {len(frame):,} records.')

    # Step 4: Run Batch Predictions
    with st.container(border=True):
        st.markdown('##### 4 · Execute Pipeline Inference')
        st.write(f'Active Pipeline: **{selected.replace("_", " ")}**. Imputation and standard scaling will be performed in-memory.')
        if st.button('Analyze batch', type='primary', icon=':material/analytics:', disabled=pipeline is None, use_container_width=True):
            try:
                with st.spinner('Executing batch scoring pipeline…'):
                    results, prepared = batch_results(pipeline, frame, selected)
                st.session_state.batch_result = (signature, results, int(prepared.isna().sum().sum()))
            except (ValueError, TypeError, RuntimeError) as exc:
                st.session_state.pop('batch_result', None)
                st.error(f'Batch prediction failed: {exc}')

    saved = st.session_state.get('batch_result')
    if pipeline is None or saved is None or saved[0] != signature:
        return

    _, results, missing_count = saved

    # Step 5: Review & Download Results
    st.markdown('---')
    st.markdown('### 5 · Batch Prediction Results & Export')
    metric_row([
        ('Samples analyzed', f'{len(results):,}', 'Total records processed in batch run.'),
        ('0 = Not Potable', f'{results.Predicted_Class.eq(0).sum():,}', 'Predicted dataset class 0.'),
        ('1 = Potable', f'{results.Predicted_Class.eq(1).sum():,}', 'Predicted dataset class 1.'),
        ('Missing input cells', f'{missing_count:,}', 'Includes zero pH values; imputed internally by Pipeline.'),
    ])

    st.dataframe(results, width='stretch', hide_index=True)
    st.caption('Original identifiers and metadata columns are preserved. Input parameters maintain their uploaded values.')
    threshold = decision_threshold(pipeline)
    if threshold is not None:
        st.caption(f'Decision rule: Potable is predicted only if P(Potable) ≥ {threshold:.0%}; the probability columns are the model’s raw outputs.')

    st.download_button('Download batch results', results.to_csv(index=False).encode('utf-8'),
                       'batch_predictions.csv', mime='text/csv', type='primary', icon=':material/download:', use_container_width=True)
    st.warning(DISCLAIMER)

    if 'Max_Class_Probability' in results:
        chart(px.histogram(results, x='Max_Class_Probability', nbins=20,
                           title='Distribution of highest class probabilities',
                           labels={'Max_Class_Probability': 'Highest model probability (%)'},
                           color_discrete_sequence=['#08949C']), 'batch_probability', 320, dark_mode)


def render_dataset(data, dark_mode: bool = False):
    page_header('Dataset Explorer', 'Exploratory data analysis of water_potability.csv: distributions, missingness, and feature correlations.', 'Quality Audit')
    if data is None:
        st.error('Dataset unavailable. Restore dataset/water_potability.csv to use this view.')
        return

    # 1. Dataset overview and KPI metrics
    metric_row([
        ('Dataset records', f'{len(data):,}', 'Original dataset rows.'),
        ('Input features', len(FEATURES), 'The Potability target is not an input feature.'),
        ('Missing feature cells', f'{data[FEATURES].isna().sum().sum():,}', 'Raw CSV missing values; excludes zero pH.'),
        ('Zero pH records', int(data.ph.eq(0).sum()), 'Treated as missing only during preprocessing and prediction.'),
    ])

    st.markdown('---')

    # 2 & 3. Target-class distribution & Missing-value analysis (balanced side-by-side)
    left, right = st.columns(2, gap='large')
    with left:
        with st.container(border=True):
            st.markdown('#### Target-Class Distribution')
            chart(distribution_chart(data, dark_mode), 'dataset_classes', 340, dark_mode)
            st.caption('Historical recorded target classes. Class 0: Not Potable (61.0%) vs Class 1: Potable (39.0%).')

    with right:
        missing = pd.DataFrame({'Feature': FEATURES, 'Missing cells': data[FEATURES].isna().sum().values})
        missing['Missing (%)'] = missing['Missing cells'] / len(data) * 100
        with st.container(border=True):
            st.markdown('#### Missing-Value Analysis (Raw CSV)')
            chart(px.bar(missing.sort_values('Missing cells'), x='Missing cells', y='Feature', orientation='h',
                         title='Missing observations per parameter', color_discrete_sequence=['#6C8797']), 'dataset_missing', 240, dark_mode)
            st.dataframe(missing.sort_values('Missing cells', ascending=False).reset_index(drop=True), width='stretch', hide_index=True,
                         column_config={'Missing cells': st.column_config.NumberColumn(format='%d'), 'Missing (%)': st.column_config.NumberColumn(format='%.1f%%')})

    st.markdown('---')

    # 4 & 5. Feature selection & Feature distribution and exploration
    with st.container(border=True):
        st.markdown('#### Feature Exploration & Class Separation')
        feature = st.selectbox('Explore a feature', FEATURES, format_func=lambda name: FIELD_SPECS[name][0], key='dataset_feature')

        plot_data = data[[feature, 'Potability']].copy()
        plot_data['Target class'] = plot_data.Potability.map(CLASS_LABELS)
        color_map = COLORS_DARK if dark_mode else COLORS_LIGHT
        fig = px.histogram(plot_data, x=feature, color='Target class', nbins=35, barmode='overlay', opacity=.75,
                           category_orders={'Target class': list(CLASS_LABELS.values())},
                           color_discrete_map=color_map, title=f'{FIELD_SPECS[feature][0]} by recorded target class',
                           labels={feature: FIELD_SPECS[feature][0]})
        fig.update_yaxes(title='Number of records')
        chart(fig, 'dataset_distribution', 360, dark_mode)
        st.caption('Raw observations are plotted without imputation. Missing values are omitted; zero pH records remain visible here.')

    st.markdown('---')

    # Advanced Exploration: Correlation heatmap & Summary statistics
    st.markdown('#### Multivariate Correlation & Descriptive Statistics')
    left_c, right_c = st.columns([1.2, 1], gap='large')
    with left_c:
        with st.container(border=True):
            st.markdown('##### Pairwise Pearson Correlation')
            corr = data[FEATURES].corr()
            scale = ['#1E293B', '#0B131E', '#38E1E6'] if dark_mode else ['#29485D', '#F7FAFB', '#08949C']
            fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1,
                            color_continuous_scale=scale,
                            title='Feature Correlation Matrix', aspect='auto',
                            labels={'x': 'Feature', 'y': 'Feature', 'color': 'Correlation'})
            chart(fig, 'dataset_correlations', 440, dark_mode)
            st.caption('Pairwise complete observations; correlation is not evidence of causation or potability.')
    with right_c:
        with st.container(border=True):
            st.markdown('##### Descriptive Summary Statistics')
            st.dataframe(data.describe().T[['count', 'mean', 'std', 'min', 'max']], width='stretch',
                         column_config={c: st.column_config.NumberColumn(format='%.2f') for c in ['mean', 'std', 'min', 'max']})
            with st.expander('Inspect Raw Data Samples (Top 50 Rows)'):
                st.dataframe(data.head(50), width='stretch', hide_index=True)


def render_performance(pipeline, evaluations, selected, dark_mode: bool = False):
    page_header('Model Performance', 'Benchmark metrics from 5×5 nested cross-validation and held-out test split (80/20).', 'Benchmark Report')
    result = evaluations.get(selected)
    if not result:
        st.info('No evaluation report is available for this model. Run python train_model.py to generate it.')
        return

    # 1. Headline test and validation metrics
    metric_row([
        ('Test accuracy', metric_value(result, 'test_accuracy'), 'Correct classifications on the untouched test split.'),
        ('Test Macro F1', metric_value(result, 'test_macro_f1'), 'F1 averaged equally over both target classes.'),
        ('Validation Macro F1', metric_value(result, 'cv_macro_f1_mean'), 'Five-fold mean used for model selection.'),
        ('Test weighted F1', metric_value(result, 'test_weighted_f1'), 'F1 averaged by class frequency.'),
    ])

    # 2. Overfitting warning and train-validation gap
    overfit_notice(result)

    st.markdown('---')

    # 3 & 4. Model comparison chart and detailed model comparison table
    with st.container(border=True):
        st.markdown('#### Model Comparison: Validation vs Test Macro F1')
        comparison = pd.DataFrame([
            {'Model': name.replace('_', ' '), 'Validation Macro F1': r.get('cv_macro_f1_mean'),
             'Test Macro F1': r.get('test_macro_f1'), 'Test accuracy': r.get('test_accuracy'),
             'Test weighted F1': r.get('test_weighted_f1'),
             'CV Macro F1 std': r.get('cv_macro_f1_std')}
            for name, r in evaluations.items()
        ]).sort_values('Validation Macro F1', ascending=False)

        bar_colors = ['#475569', '#38E1E6'] if dark_mode else ['#29485D', '#08949C']
        fig = px.bar(comparison.melt(id_vars='Model', value_vars=['Validation Macro F1', 'Test Macro F1'],
                                     var_name='Evaluation', value_name='Score'),
                     x='Score', y='Model', color='Evaluation', orientation='h', barmode='group',
                     title='Model Comparison · Validation vs Test Macro F1', color_discrete_sequence=bar_colors)
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(autorange='reversed')
        chart(fig, 'performance_comparison', 380, dark_mode)

        st.markdown('##### Detailed Model Comparison Metrics')
        st.dataframe(comparison, width='stretch', hide_index=True,
                     column_config={name: st.column_config.NumberColumn(format='%.4f')
                                    for name in comparison.columns if name != 'Model'})
        st.caption('Baseline Majority always predicts the most frequent training class. Model selection uses validation Macro F1, never test scores.')

    st.markdown('---')

    # 5 & 6. Confusion matrix and per-class performance (side-by-side)
    left, right = st.columns(2, gap='large')
    with left:
        with st.container(border=True):
            st.markdown('#### Confusion Matrix')
            cm = result.get('confusion_matrix')
            if cm is not None and np.asarray(cm).shape == (2, 2):
                scale = ['#162032', '#087E8B'] if dark_mode else ['#EDF3F5', '#087E8B']
                fig = go.Figure(go.Heatmap(z=cm, x=list(CLASS_LABELS.values()), y=list(CLASS_LABELS.values()),
                                          text=cm, texttemplate='%{text}', showscale=False,
                                          colorscale=scale,
                                          hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Records: %{z}<extra></extra>'))
                fig.update_layout(title=f'Confusion Matrix · {selected.replace("_", " ")}',
                                  xaxis_title='Predicted class (columns)', yaxis_title='Actual class (rows)')
                fig.update_yaxes(autorange='reversed')
                chart(fig, 'performance_confusion', 340, dark_mode)
            else:
                st.info('Confusion matrix unavailable.')
            st.caption('Rows = actual classes. Columns = predicted classes. 0 = Not Potable · 1 = Potable.')

    with right:
        with st.container(border=True):
            st.markdown('#### Per-Class Performance & Error Breakdown')
            per_class = pd.DataFrame([
                {'Target class': CLASS_LABELS[label],
                 'Precision': result.get(f'test_precision_class_{label}'),
                 'Recall': result.get(f'test_recall_class_{label}'),
                 'F1': result.get(f'test_f1_class_{label}')}
                for label in (0, 1)
            ])
            st.dataframe(per_class, width='stretch', hide_index=True,
                         column_config={name: st.column_config.NumberColumn(format='%.3f') for name in ['Precision', 'Recall', 'F1']})
            st.caption('Precision: proportion of positive identifications that were correct. Recall: proportion of actual positives identified correctly. F1: harmonic mean.')
            if cm is not None and np.asarray(cm).shape == (2, 2):
                st.warning(f'{cm[0][1]:,} test records labeled 0 = Not Potable were predicted as 1 = Potable. This is a critical error mode requiring caution.')

    st.markdown('---')

    # 7. Generalization and Feature Importance (if available)
    st.markdown('#### Generalization Dynamics & Feature Drivers')
    has_importance = False
    importance, label = None, None
    if pipeline is not None:
        importance, label = feature_importance(pipeline)
        if importance is not None:
            has_importance = True

    if has_importance:
        c_left, c_right = st.columns(2, gap='large')
        with c_left:
            with st.container(border=True):
                st.markdown('##### Training vs Unseen Data Accuracy')
                stages = pd.DataFrame({'Evaluation stage': ['Fold training (mean)', 'Fold validation (mean)', 'Held-out test'],
                                       'Accuracy': [result.get('train_accuracy_mean'), result.get('cv_accuracy_mean'), result.get('test_accuracy')],
                                       'Standard deviation': [result.get('train_accuracy_std', 0), result.get('cv_accuracy_std', 0), 0]})
                fig = px.bar(stages, x='Evaluation stage', y='Accuracy', error_y='Standard deviation',
                             title='Training vs Unseen Data Accuracy', color_discrete_sequence=['#475569' if dark_mode else '#29485D'])
                fig.update_yaxes(range=[0, 1.05])
                chart(fig, 'performance_overfit', 320, dark_mode)
                st.caption('Error bars show standard deviation across CV folds. Test accuracy is an untouched single evaluation.')
        with c_right:
            with st.container(border=True):
                st.markdown(f'##### {label}')
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                             title=f'{label} · {selected.replace("_", " ")}', color_discrete_sequence=['#38E1E6' if dark_mode else '#08949C'])
                fig.update_xaxes(title=label)
                chart(fig, 'performance_importance', 320, dark_mode)
                st.caption('Global model associations, not a causal explanation or laboratory drinking-water threshold.')
    else:
        with st.container(border=True):
            st.markdown('##### Training vs Unseen Data Accuracy')
            stages = pd.DataFrame({'Evaluation stage': ['Fold training (mean)', 'Fold validation (mean)', 'Held-out test'],
                                   'Accuracy': [result.get('train_accuracy_mean'), result.get('cv_accuracy_mean'), result.get('test_accuracy')],
                                   'Standard deviation': [result.get('train_accuracy_std', 0), result.get('cv_accuracy_std', 0), 0]})
            fig = px.bar(stages, x='Evaluation stage', y='Accuracy', error_y='Standard deviation',
                         title='Training vs Unseen Data Accuracy', color_discrete_sequence=['#475569' if dark_mode else '#29485D'])
            fig.update_yaxes(range=[0, 1.05])
            chart(fig, 'performance_overfit', 320, dark_mode)
            st.caption('Error bars show standard deviation across CV folds. Test accuracy is an untouched single evaluation.')

    st.markdown('---')

    # Threshold & Downloads
    threshold = decision_threshold(pipeline)
    if threshold is not None:
        st.caption(f'Decision rule: Potable is predicted only if P(Potable) ≥ {threshold:.0%}; the probability columns are the model’s raw outputs.')
    with st.container(horizontal=True):
        for filename, label_text, mime in [('evaluation_report.txt', 'Download evaluation report', 'text/plain'),
                                           ('evaluation_results.json', 'Download evaluation JSON', 'application/json')]:
            path = BASE_DIR / 'models' / filename
            if path.exists():
                st.download_button(label_text, path.read_bytes(), filename, mime=mime, icon=':material/download:')


def render_methodology(evaluations, selected, dark_mode: bool = False):
    page_header('Training & Methodology', 'Rigorous scientific methodology, pipeline architecture, cross-validation protocol, and model limitations.', 'Methodology Specification')

    # 1. Pipeline Architecture & Methodology Protocol
    with st.container(border=True):
        st.markdown('#### Standardized Pipeline Protocol')
        st.markdown('''
        **01 / Dataset Acquisition & Scope**  
        `dataset/water_potability.csv` supplies 3,276 records across nine numeric parameters and the binary target `Potability`. **0 = Not Potable** (61.0%) and **1 = Potable** (39.0%) represent historical database labels. Original CSV records are never overwritten.

        **02 / Missing-Value Handling & Zero pH Convention**  
        The project treats zero pH as a missing-measurement indicator via `treat_zero_ph_as_missing`. Imputation is performed inside the pipeline via training-fold medians (`SimpleImputer(strategy="median")`), preventing data leakage.

        **03 / Stratified Partitioning**  
        A stratified **80% training / 20% test** split is established using `random_state=42`. The test set remains untouched and is evaluated exactly once after all tuning is finalized.

        **04 / Pipeline Preprocessing & Scaling**  
        Each pipeline chains `SimpleImputer(strategy="median") → StandardScaler() → Classifier`. Preprocessing parameters are fitted strictly on training folds and applied to validation/test partitions.

        **05 / 5×5 Nested Cross-Validation**  
        Nested cross-validation (5 outer folds × 5 inner folds) is employed for hyperparameter optimization and unbiased performance estimation across Random Forest, SVC, Extra Trees, XGBoost, and Decision Tree. Paired corrected t-tests verify statistical significance.

        **06 / Precision-Targeted Decision Threshold Calibration**  
        To minimize hazardous false-potable classifications, decision thresholds are calibrated on out-of-fold training predictions to achieve precision ≥ 0.70 where attainable.

        **07 / Held-Out Test Evaluation**  
        Final pipelines are retrained on the full training split with optimal hyperparameters and threshold rules, then evaluated on the untouched test partition.
        ''')

    # 2. Model Zoo Summary Table
    with st.container(border=True):
        st.markdown('#### Evaluated Machine Learning Pipelines')
        model_zoo = pd.DataFrame([
            {'Pipeline': 'Random Forest', 'Family': 'Ensemble (Bagging)', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'n_estimators, max_depth, min_samples_split'},
            {'Pipeline': 'SVC (RBF)', 'Family': 'Support Vector Machine', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'C, gamma, kernel'},
            {'Pipeline': 'Extra Trees', 'Family': 'Ensemble (Extremely Randomized)', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'n_estimators, max_depth, min_samples_split'},
            {'Pipeline': 'Soft Voting (RF + SVC)', 'Family': 'Ensemble (Voting)', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'Weighted average of calibrated probabilities'},
            {'Pipeline': 'XGBoost', 'Family': 'Gradient Boosting', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'scale_pos_weight, learning_rate, max_depth'},
            {'Pipeline': 'Logistic Regression', 'Family': 'Linear Model', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Out-of-fold Precision ≥ 0.70', 'Tuned Hyperparameters': 'C, penalty, solver'},
            {'Pipeline': 'Decision Tree', 'Family': 'Single Tree', 'Preprocessing': 'Median Imputer + StandardScaler', 'Threshold Rule': 'Standard Rule (0.50)', 'Tuned Hyperparameters': 'max_depth, min_samples_leaf'},
            {'Pipeline': 'Baseline Majority', 'Family': 'Heuristic Baseline', 'Preprocessing': 'None', 'Threshold Rule': 'Always predicts class 0', 'Tuned Hyperparameters': 'None (Reference benchmark)'},
        ])
        st.dataframe(model_zoo, width='stretch', hide_index=True)

    # 3. Limitations & Boundary Conditions
    with st.container(border=True):
        st.markdown('#### Known Limitations & Scientific Boundary Conditions')
        overfit_notice(evaluations.get(selected, {}))
        st.markdown('''
        - **Imperfect Generalization:** Overall test accuracy (~65–68%) masks significant class-specific error rates. Models cannot guarantee absence of contaminants.
        - **Data Scope Limitations:** The dataset contains only nine chemical parameters. It excludes microbiological pathogens (E. coli, coliforms), heavy metals (lead, arsenic), microplastics, and radiological indicators.
        - **Small Statistical Margins:** Score differences between leading ensemble models fall within one cross-validation standard deviation.
        - **Threshold Trade-offs:** Raising the potable probability threshold improves precision but reduces recall, rejecting genuine potable samples.
        - **Non-Causality of Feature Importance:** Feature importance reflects mathematical splits, not toxicological causation or safe consumption thresholds.
        - **Educational Prototype:** This system is an educational demonstrator. It must never be used for municipal, clinical, or domestic drinking-water safety clearance.
        ''')

    st.warning(DISCLAIMER)

    # 4. Local Execution & Reproducibility
    with st.expander('Local Reproduction & CLI Verification Commands'):
        st.code('python train_model.py\npython test_models.py\npython test_app.py\npython -m streamlit run app.py', language='bash')
        st.caption('Training executes the full 5×5 nested cross-validation pipeline, generates serialized artifacts in models/, and exports evaluation reports.')


def main():
    # Sidebar theme toggle & navigation
    with st.sidebar:
        st.markdown('''
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon">💧</div>
          <div>
            <div class="sidebar-brand-title">Water Quality</div>
            <div class="sidebar-brand-subtitle">Intelligence Platform</div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        view = init_sidebar_nav()

        # Appearance / Theme Switcher in Sidebar
        st.markdown('''
        <div class="sidebar-model-section">
          <div class="sidebar-section-label">APPEARANCE</div>
        </div>
        ''', unsafe_allow_html=True)
        dark_mode = st.toggle('🌙 Dark Mode', key='theme_dark', value=st.session_state.get('theme_dark', False))

        # Model Selector placed in a clearly separated section
        st.markdown('''
        <div class="sidebar-model-section">
          <div class="sidebar-section-label">ACTIVE PIPELINE</div>
        </div>
        ''', unsafe_allow_html=True)

    # Apply CSS dynamically based on chosen theme
    st.markdown(get_css(dark_mode), unsafe_allow_html=True)

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
        selected = st.selectbox('Prediction model', MODEL_NAMES,
                                index=MODEL_NAMES.index(recommended) if recommended else 0,
                                format_func=lambda name: name.replace('_', ' '),
                                key='selected_model',
                                label_visibility='collapsed')

        if recommended and selected == recommended:
            st.markdown('<div class="model-badge">★ Recommended (Highest CV Macro F1)</div>', unsafe_allow_html=True)
        elif recommended:
            st.caption(f'Recommended: {recommended.replace("_", " ")}')

    # Load pipelines
    models, model_errors = {}, {}
    schema_version = artifact_version(BASE_DIR / 'models' / 'feature_names.pkl')
    for name in MODEL_NAMES:
        try:
            models[name] = cached_pipeline(name, artifact_version(BASE_DIR / 'models' / f'{name}.pkl'), schema_version)
        except Exception as exc:
            model_errors[name] = str(exc)

    pipeline = models.get(selected)

    # Top App Bar with visibility fix and theme badge
    render_app_bar(selected, pipeline, recommended, len(models), dark_mode)

    # Global alerts
    for error in errors:
        st.warning(error)
    if model_errors:
        with st.expander('Model loading details', expanded=pipeline is None):
            for name, error in model_errors.items():
                st.error(f'{name}: {error}')
            st.code('python train_model.py', language='bash')

    # View Router
    version = (artifact_version(BASE_DIR / 'models' / f'{selected}.pkl'), schema_version)
    if view == 'Overview':
        render_overview(data, evaluations, selected, recommended, dark_mode)
    elif view == 'Water Prediction':
        render_prediction(data, pipeline, selected, version, dark_mode)
    elif view == 'Batch Analysis':
        render_batch(pipeline, selected, dark_mode)
    elif view == 'Dataset Explorer':
        render_dataset(data, dark_mode)
    elif view == 'Model Performance':
        render_performance(pipeline, evaluations, selected, dark_mode)
    else:
        render_methodology(evaluations, selected, dark_mode)

    # Standard Disclaimer in every view
    st.info(DISCLAIMER, icon=':material/info:')

    # Modern Minimalist Footer
    st.markdown('''
    <div class="app-footer">
      <strong>Water Quality Intelligence Platform</strong> · Dataset-based Potability Classification<br>
      Python · Pandas · NumPy · Scikit-learn Pipelines · XGBoost · Streamlit · Plotly<br>
      Educational & Research Prototype · Not a drinking-water safety certification or clinical diagnostic tool.
    </div>
    ''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
