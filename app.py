
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Water Quality Predictor", page_icon="💧", layout="wide")

INPUT_FIELDS = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
    'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
INPUT_WIDGET_KEYS = {
    'ph': 'ph_input_widget',
    'Hardness': 'hardness_input_widget',
    'Solids': 'solids_input_widget',
    'Chloramines': 'chloramines_input_widget',
    'Sulfate': 'sulfate_input_widget',
    'Conductivity': 'conductivity_input_widget',
    'Organic_carbon': 'organic_carbon_input_widget',
    'Trihalomethanes': 'trihalomethanes_input_widget',
    'Turbidity': 'turbidity_input_widget',
}

def set_input_values(values):
    """Set both the saved inputs and the number-input widget state.

    Streamlit widgets retain state by their key, so updating only ``inputs``
    does not change what the user sees or what is predicted.
    """
    st.session_state.inputs = {name: float(values[name]) for name in INPUT_FIELDS}
    for name, widget_key in INPUT_WIDGET_KEYS.items():
        st.session_state[widget_key] = st.session_state.inputs[name]

st.markdown("""
    <style>
    .main {
        background-color: #f0f7ff;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_model_and_scaler(model_name='Random_Forest'):
    model_path = f'models/{model_name}.pkl'
    scaler_path = 'models/scaler.pkl'
    imputer_path = 'models/imputer.pkl'
    feature_names_path = 'models/feature_names.pkl'
    
    if (not os.path.exists(model_path) or 
        not os.path.exists(scaler_path) or 
        not os.path.exists(feature_names_path) or
        not os.path.exists(imputer_path)):
        st.error("Model files not found! Please train the models first using train_model.py.")
        return None, None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    feature_names = joblib.load(feature_names_path)
    return model, scaler, imputer, feature_names

def predict_water_quality(model, scaler, imputer, features, feature_names):
    features_df = pd.DataFrame([features], columns=feature_names)
    features_imputed = imputer.transform(features_df)
    features_scaled = scaler.transform(features_imputed)
    
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    confidence = max(prediction_proba) * 100
    is_safe = prediction == 1
    
    return is_safe, confidence, prediction_proba, features_df

def sample_with_prediction(samples, target_class, model, scaler, imputer, feature_names):
    """Return a dataset row whose current model prediction matches the button."""
    if samples is None or samples.empty:
        return None

    candidate_features = samples.loc[:, feature_names]
    scaled = scaler.transform(imputer.transform(candidate_features))
    matching = samples.loc[model.predict(scaled) == target_class]
    return None if matching.empty else matching.sample(1).iloc[0]

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def get_rule_based_warnings(feature_values):
    """Generate rule-based warnings based on WHO guidelines."""
    warnings = []
    
    if feature_values['ph'] < 6.5 or feature_values['ph'] > 8.5:
        warnings.append("⚠ pH is outside the WHO recommended drinking water range (6.5-8.5).")
    
    if feature_values['Turbidity'] > 5:
        warnings.append("⚠ High turbidity detected. WHO recommends turbidity < 5 NTU.")
    
    if feature_values['Chloramines'] > 4:
        warnings.append("⚠ Chloramine levels are above the EPA recommended limit of 4 mg/L.")
    
    if feature_values['Sulfate'] > 250:
        warnings.append("⚠ Sulfate levels are above the WHO recommended limit of 250 mg/L.")
    
    return warnings

def create_parameter_chart(features, feature_names):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_names,
        y=features,
        marker_color='#1f77b4',
        opacity=0.8
    ))
    
    fig.update_layout(
        title='Water Quality Parameters',
        xaxis_title='Parameters',
        yaxis_title='Values',
        xaxis_tickangle=-45,
        template='plotly_white'
    )
    
    return fig

def create_importance_chart(importance_df):
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='viridis',
        template='plotly_white'
    )
    return fig

def generate_report(features, feature_names, is_safe, confidence, probs, warnings):
    report = []
    report.append("=" * 60)
    report.append("WATER QUALITY PREDICTION REPORT")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    report.append("INPUT PARAMETERS:")
    for name, value in zip(feature_names, features):
        report.append(f"  {name}: {value:.2f}")
    report.append("")
    report.append("PREDICTION RESULTS:")
    report.append(f"  Water Quality: {'SAFE' if is_safe else 'UNSAFE'} for drinking")
    report.append(f"  Probability UNSAFE: {probs[0]:.1%}")
    report.append(f"  Probability SAFE: {probs[1]:.1%}")
    report.append(f"  Confidence Score: {confidence:.1f}%")
    report.append("")
    if warnings:
        report.append("RULE-BASED WARNINGS:")
        for warn in warnings:
            report.append(f"  - {warn}")
    report.append("")
    report.append("=" * 60)
    return "\n".join(report)

def main():
    st.title("💧 AI-Based Water Quality Prediction")
    st.markdown("---")
    
    st.sidebar.title("🔧 Settings")
    model_option = st.sidebar.selectbox(
        "Select ML Model",
        ['Random_Forest', 'XGBoost', 'Decision_Tree', 'Logistic_Regression'],
        index=0
    )
    
    # Load dataset for sample buttons!
    data_path = 'dataset/water_potability.csv'
    df = None
    safe_samples = None
    unsafe_samples = None
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        safe_samples = df[df['Potability'] == 1].dropna()
        unsafe_samples = df[df['Potability'] == 0].dropna()
    
    # Initialize session state if not already initialized
    if 'inputs' not in st.session_state:
        if safe_samples is not None and not safe_samples.empty:
            s = safe_samples.iloc[0]
            st.session_state.inputs = {
                'ph': s['ph'], 'Hardness': s['Hardness'],
                'Solids': s['Solids'], 'Chloramines': s['Chloramines'],
                'Sulfate': s['Sulfate'], 'Conductivity': s['Conductivity'],
                'Organic_carbon': s['Organic_carbon'],
                'Trihalomethanes': s['Trihalomethanes'],
                'Turbidity': s['Turbidity']
            }
        else:
            st.session_state.inputs = {
                'ph':7.0, 'Hardness':196.0, 'Solids':20000.0,
                'Chloramines':7.0, 'Sulfate':333.0, 'Conductivity':420.0,
                'Organic_carbon':14.0, 'Trihalomethanes':66.0, 'Turbidity':4.0
            }
    
    # Load before creating example buttons, so their labels match the model's
    # prediction instead of only the historical dataset label.
    model, scaler, imputer, feature_names = load_model_and_scaler(model_option)

    st.sidebar.markdown("### 📋 Example Inputs")
    
    # Button 1: Safe Sample
    if st.sidebar.button("🟢 Safe Sample", use_container_width=True, key='safe_btn'):
        if model is not None:
            s = sample_with_prediction(safe_samples, 1, model, scaler, imputer, feature_names)
        else:
            s = None
        if s is not None:
            set_input_values(s)
            st.sidebar.success("Loaded a safe sample!")
        else:
            st.sidebar.error("No dataset sample is currently predicted safe by this model.")
    
    # Button 2: Unsafe Sample
    if st.sidebar.button("🔴 Unsafe Sample", use_container_width=True, key='unsafe_btn'):
        if model is not None:
            u = sample_with_prediction(unsafe_samples, 0, model, scaler, imputer, feature_names)
        else:
            u = None
        if u is not None:
            set_input_values(u)
            st.sidebar.success("Loaded an unsafe sample!")
        else:
            st.sidebar.error("No dataset sample is currently predicted unsafe by this model.")
    
    # Button 3: Random Sample
    if st.sidebar.button("🎲 Random Sample", use_container_width=True, key='random_btn'):
        if df is not None:
            r = df.dropna().sample(1).iloc[0]
            set_input_values(r)
            st.sidebar.success("Loaded a random sample!")
        else:
            st.sidebar.error("No dataset available")
    
    if model is None:
        st.warning("Please train the models first! Check the 'Training Instructions' section below.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Explorer", "🎯 Manual Prediction", "📈 Batch Prediction", "📋 Training Info"])
    
    with tab1:
        st.subheader("📊 Dataset Explorer")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.write("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("### Dataset Statistics")
                st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.write("### Potability Distribution")
                fig_dist = px.pie(df, names='Potability', 
                                title='Water Potability (0 = Unsafe, 1 = Safe)',
                                color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
                st.plotly_chart(fig_dist, use_container_width=True)
            
            st.write("### Correlation Heatmap")
            corr = df.corr()
            fig_corr = px.imshow(corr, 
                                 text_auto=True, 
                                 aspect="auto",
                                 color_continuous_scale='RdBu_r',
                                 title='Feature Correlation Matrix')
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.write("### Feature Distributions")
            selected_feature = st.selectbox("Select Feature to Visualize", df.columns[:-1])
            fig_hist = px.histogram(df, x=selected_feature, color='Potability',
                                    title=f'Distribution of {selected_feature} by Potability',
                                    color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.error("Dataset not found! Please ensure 'dataset/water_potability.csv' exists.")
    
    with tab2:
        st.subheader("🎯 Enter Water Quality Parameters")
        st.markdown("Enter the parameters below to predict water quality:")
        
        col1, col2, col3 = st.columns(3)
        
        param_info = {
            'ph': "pH level (WHO recommended: 6.5-8.5)",
            'Hardness': "Calcium and magnesium concentration",
            'Solids': "Total dissolved solids (TDS)",
            'Chloramines': "Chlorine and ammonia compound (EPA limit: 4 mg/L)",
            'Sulfate': "Sulfate ion concentration (WHO limit: 250 mg/L)",
            'Conductivity': "Electrical conductivity",
            'Organic_carbon': "Total organic carbon",
            'Trihalomethanes': "Disinfection byproducts",
            'Turbidity': "Cloudiness of water (WHO limit: <5 NTU)"
        }
        
        # Input fields using session state values
        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, 
                                 value=st.session_state.inputs['ph'], 
                                 step=0.1, help=param_info['ph'], key='ph_input_widget')
            hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, 
                                       value=st.session_state.inputs['Hardness'], 
                                       step=1.0, help=param_info['Hardness'], key='hardness_input_widget')
            solids = st.number_input("Solids (TDS)", min_value=0.0, max_value=50000.0, 
                                     value=st.session_state.inputs['Solids'], 
                                     step=10.0, help=param_info['Solids'], key='solids_input_widget')
        
        with col2:
            chloramines = st.number_input("Chloramines", min_value=0.0, max_value=20.0, 
                                          value=st.session_state.inputs['Chloramines'], 
                                          step=0.1, help=param_info['Chloramines'], key='chloramines_input_widget')
            sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, 
                                      value=st.session_state.inputs['Sulfate'], 
                                      step=1.0, help=param_info['Sulfate'], key='sulfate_input_widget')
            conductivity = st.number_input("Conductivity", min_value=0.0, max_value=1000.0, 
                                           value=st.session_state.inputs['Conductivity'], 
                                           step=1.0, help=param_info['Conductivity'], key='conductivity_input_widget')
        
        with col3:
            organic_carbon = st.number_input("Organic Carbon", min_value=0.0, max_value=30.0, 
                                             value=st.session_state.inputs['Organic_carbon'], 
                                             step=0.1, help=param_info['Organic_carbon'], key='organic_carbon_input_widget')
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=200.0, 
                                              value=st.session_state.inputs['Trihalomethanes'], 
                                              step=0.1, help=param_info['Trihalomethanes'], key='trihalomethanes_input_widget')
            turbidity = st.number_input("Turbidity", min_value=0.0, max_value=10.0, 
                                        value=st.session_state.inputs['Turbidity'], 
                                        step=0.1, help=param_info['Turbidity'], key='turbidity_input_widget')
        
        # Update session state whenever inputs change
        current_inputs = {
            'ph': ph, 'Hardness': hardness,
            'Solids': solids, 'Chloramines': chloramines,
            'Sulfate': sulfate, 'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }
        if current_inputs != st.session_state.inputs:
            st.session_state.inputs = current_inputs
        
        if st.button("Predict Water Quality", type="primary", key='predict_btn'):
            is_safe, confidence, probs, features_df = predict_water_quality(
                model, scaler, imputer, [ph, hardness, solids, chloramines, sulfate, 
                                         conductivity, organic_carbon, trihalomethanes, turbidity], 
                feature_names
            )
            
            # Get rule-based warnings!
            rule_warnings = get_rule_based_warnings(current_inputs)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Prediction Result")
                if is_safe:
                    st.success(f"✅ Water Quality: SAFE for drinking")
                else:
                    st.error(f"❌ Water Quality: UNSAFE for drinking")
                
                st.info(f"Confidence Score: {confidence:.1f}%")
                st.write(f"**Probability UNSAFE:** {probs[0]:.1%}")
                st.write(f"**Probability SAFE:** {probs[1]:.1%}")
                
                if rule_warnings:
                    st.markdown("### Rule-Based Warnings (WHO/EPA Guidelines)")
                    for warn in rule_warnings:
                        st.warning(warn)
                
                report = generate_report(
                    [ph, hardness, solids, chloramines, sulfate, 
                     conductivity, organic_carbon, trihalomethanes, turbidity], 
                    feature_names, is_safe, confidence, probs, rule_warnings
                )
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"water_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                st.markdown("### Parameter Visualization")
                fig = create_parameter_chart(
                    [ph, hardness, solids, chloramines, sulfate, 
                     conductivity, organic_carbon, trihalomethanes, turbidity], 
                    feature_names
                )
                st.plotly_chart(fig, use_container_width=True)
            
            importance_df = get_feature_importance(model, feature_names)
            if importance_df is not None:
                st.markdown("### Feature Importance")
                fig_importance = create_importance_chart(importance_df)
                st.plotly_chart(fig_importance, use_container_width=True)
            
    with tab3:
        st.subheader("📈 Batch Prediction from CSV")
        st.write("Upload a CSV file with the same columns as the training dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.write("### Preview of uploaded data:")
                st.dataframe(df.head(), use_container_width=True)
                
                if set(feature_names).issubset(df.columns):
                    df_features = df[feature_names]
                    df_imputed = imputer.transform(df_features)
                    df_scaled = scaler.transform(df_imputed)
                    
                    predictions = model.predict(df_scaled)
                    prediction_probas = model.predict_proba(df_scaled)
                    
                    df['Prediction'] = ['SAFE' if pred == 1 else 'UNSAFE' for pred in predictions]
                    df['Confidence'] = [max(prob) * 100 for prob in prediction_probas]
                    df['Prob_UNSAFE'] = [prob[0] * 100 for prob in prediction_probas]
                    df['Prob_SAFE'] = [prob[1] * 100 for prob in prediction_probas]
                    
                    st.markdown("### Batch Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("### Prediction Summary")
                        fig = px.pie(df, names='Prediction', 
                                     title='Distribution of Predictions',
                                     color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Confidence Distribution")
                        fig_conf = px.histogram(df, x='Confidence', 
                                               title='Prediction Confidence Levels',
                                               color_discrete_sequence=['#4ecdc4'])
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"CSV file must contain these columns: {', '.join(feature_names)}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab4:
        st.subheader("📋 Training & Model Info")
        st.markdown("""
        ### Models Available:
        - **Random Forest**: Best performing model with class weighting
        - **XGBoost**: Gradient boosting with scale_pos_weight for imbalance
        - **Decision Tree**: Simple, interpretable model
        - **Logistic Regression**: Baseline classification model
        
        ### Features Used (in order):
        1. ph
        2. Hardness
        3. Solids
        4. Chloramines
        5. Sulfate
        6. Conductivity
        7. Organic_carbon
        8. Trihalomethanes
        9. Turbidity
        
        ### Improvements Made:
        - Added missing value imputation using median
        - Added class_weight='balanced' to handle class imbalance
        - Added stratify=y in train_test_split
        - Added rule-based warnings from WHO/EPA guidelines
        - Added example input buttons for demos
        
        ### Training Instructions:
        1. Ensure your dataset is at `dataset/water_potability.csv`
        2. Run the training script: `python train_model.py`
        3. The trained models will be saved in the `models` folder
        """)

if __name__ == "__main__":
    main()
