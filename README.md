# AI Water Quality Predictor

A Streamlit application that estimates drinking-water potability from common water-quality measurements. It includes interactive single predictions, dataset exploration, batch CSV predictions, downloadable reports, and four trained classification models.

> This project is an educational prediction tool. It does not replace laboratory testing, local regulations, or professional public-health advice.

## Features

- Manual prediction from nine water-quality parameters
- Safe, unsafe, and random example inputs
- Random Forest, XGBoost, Decision Tree, and Logistic Regression models
- Median imputation and standardized input preprocessing
- Dataset explorer with summaries, distributions, and correlations
- Batch CSV prediction and download
- Rule-based pH, turbidity, chloramine, and sulfate warnings
- Downloadable single-prediction reports

## Requirements

- Python 3.10 or later
- A dataset at `dataset/water_potability.csv`

Install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate       # Windows PowerShell / Command Prompt
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Run the app

Trained model files are already expected in `models/`. Start the web app with:

```bash
streamlit run app.py
```

Open the local address Streamlit prints in the terminal, usually `http://localhost:8501`.

## Train or retrain the models

If the `models/` folder is missing, or you replace the dataset, run:

```bash
python train_model.py
```

The script creates the following artifacts in `models/`:

- `Random_Forest.pkl`
- `XGBoost.pkl`
- `Decision_Tree.pkl`
- `Logistic_Regression.pkl`
- `scaler.pkl`
- `imputer.pkl`
- `feature_names.pkl`

## Input fields

The model uses these fields, in this order:

1. `ph`
2. `Hardness`
3. `Solids`
4. `Chloramines`
5. `Sulfate`
6. `Conductivity`
7. `Organic_carbon`
8. `Trihalomethanes`
9. `Turbidity`

Example buttons load their values directly into the number fields. The safe and unsafe examples are selected to match the current model's prediction; if the selected model has no matching dataset row, the app displays an explanatory message.

## Batch prediction CSV

Upload a CSV containing all nine input columns listed above. Extra columns are allowed and are retained in the downloaded results. The app adds:

- `Prediction`
- `Confidence`
- `Prob_UNSAFE`
- `Prob_SAFE`

Missing values in the required feature columns are filled using the median imputer saved during training.

## Project structure

```text
app.py                       Streamlit interface
train_model.py               Training and model export script
dataset/water_potability.csv Training dataset
models/                      Saved models and preprocessing artifacts
requirements.txt             Python dependencies
debug.py                     Diagnostic helper
test_specific.py             Single-input model check
```

## Notes

Potability predictions depend on the training dataset and are probabilistic. A result labelled `SAFE` means the selected model predicts class `1`; it is not a certification that the water is safe to drink. Always follow applicable water-testing standards and seek qualified advice for real-world decisions.
