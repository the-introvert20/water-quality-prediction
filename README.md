# Water Quality Intelligence

An educational Streamlit dashboard for dataset-based water-potability classification. **0 = Not Potable; 1 = Potable.** Neither class is a drinking-water safety certification. Laboratory testing and applicable water-quality standards remain essential; there is no live monitoring or sensor connection.

## Run locally

Run commands from the `water-quality-prediction` directory with Python and the dependencies installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Open `http://localhost:8501`.

To train or verify locally:

```bash
python train_model.py
python test_models.py
python test_app.py
```

Training is only needed to regenerate missing or outdated artifacts. It replaces four fitted Pipelines, `feature_names.pkl`, `evaluation_results.json`, and `evaluation_report.txt` under `models/`; it does not rewrite the original dataset. The app and legacy prediction checkers do not use standalone `scaler.pkl` or `imputer.pkl` files.

## Six workspace views

| View | What it provides |
|---|---|
| **Overview** | Selected-model test metrics, recorded class distribution, CV-based recommendation, and training–validation gap. |
| **Water Prediction** | Nine measurement inputs, unavailable-pH handling, Potable/Not Potable/random dataset examples, class probabilities, and a downloadable text report. Class-specific examples are filtered by the selected model's prediction, not guaranteed to match their historical labels. |
| **Batch Analysis** | Example CSV download, upload validation, explicit batch analysis, results table, probability histogram, and CSV export. |
| **Dataset Explorer** | Original class balance, missing and zero-pH counts, feature distributions, correlations, preview, and descriptive statistics. |
| **Model Performance** | Saved CV/test comparisons with a majority baseline, per-class metrics, confusion matrix, overfitting comparison, global feature importance or coefficients, and report/JSON downloads. |
| **Training & Methodology** | Actual preprocessing, splitting, model selection, limitations, and local commands. |

The sidebar selects Random Forest, XGBoost, Decision Tree, or Logistic Regression for both single and batch predictions. The default recommendation uses the highest mean cross-validation Macro F1, not test accuracy. Input limits are interface constraints, not scientific safety thresholds. The original pH, turbidity, chloramines, and sulfate reference checks remain available under the prediction result and in its report. They are explicitly labeled legacy project rules, not verified regulatory guideline alerts or a standards-compliance assessment.

## CSV inputs and downloads

Use a comma-separated UTF-8 CSV with these exact, case-sensitive feature names:

```text
ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
```

Values must be finite numbers or blank cells. All nine columns are required; extra columns are retained. `Potability` is optional and is never a prediction input. Empty files, duplicate headers, missing feature columns, invalid numeric cells, and existing result-column names are rejected.

`batch_predictions.csv` preserves submitted values and appends:

| Column | Meaning |
|---|---|
| `Prediction` | `Not Potable` or `Potable` |
| `Predicted_Class` | Numeric class `0` or `1` |
| `Max_Class_Probability` | Largest model class probability, **0–100 (%)** |
| `Prob_Not_Potable` | Probability assigned to class `0`, **0–100 (%)** |
| `Prob_Potable` | Probability assigned to class `1`, **0–100 (%)** |
| `Prediction_Model` | Selected model identifier |

Other downloads are `water_quality_example.csv`, `water_quality_report.txt` (submitted measurements, model, predicted class, probabilities, preprocessing, and disclaimer), and the saved evaluation text/JSON reports. Probabilities are model outputs, not guaranteed real-world confidence.

## Unchanged training methodology

- Source: the original `dataset/water_potability.csv`, with nine numeric features and the `Potability` target. Exploration shows raw values; prediction and training work on copies, leaving the original CSV unchanged.
- The shared `treat_zero_ph_as_missing` helper converts only `ph == 0` to missing in memory. This is a dataset convention, not a claim that pH zero is physically impossible.
- Stratified **80% training / 20% test**, `random_state=42`; five-fold `StratifiedKFold(shuffle=True, random_state=42)` on the training partition.
- Each candidate Pipeline fits `SimpleImputer(strategy="median") → StandardScaler → classifier` inside every CV training fold. Inference uses the fitted Pipeline without separate imputation or scaling.
- Random Forest: 200 trees, depth 10; XGBoost: 200 estimators, depth 5; Decision Tree: depth 10; Logistic Regression: maximum 1,000 iterations. The non-XGBoost candidates use balanced class weights; XGBoost uses `scale_pos_weight`. A majority-class baseline is evaluated but not saved as a prediction artifact.
- Highest mean validation Macro F1 selects the recommended candidate. Each candidate is refitted on the full **training partition**, then evaluated on the held-out test partition; saved models are not refitted on the full dataset.

## Limitations

- Large training–validation accuracy gaps indicate **overfitting**; inspect both class recalls and the confusion matrix, not accuracy alone. Not Potable records can be predicted as Potable.
- **XGBoost class-ratio caveat:** the existing training script computes `scale_pos_weight` from full-dataset target counts before splitting. Preprocessing is fold-local, but this weight uses held-out label counts. The methodology is preserved, not claimed to be fully leakage-free.
- Missing measurements, class imbalance, and limited dataset coverage constrain generalization to new locations, instruments, contaminants, and time periods. There is no external validation.
- Probabilities have not been demonstrated to be calibrated estimates of drinking-water safety. Feature importance describes model associations, not causation, approved limits, or treatment advice.

For read-only diagnostics, run `python audit_dataset.py`, `python debug.py`, or `python test_specific.py`. The latter two are legacy Random Forest prediction checkers, not model-training or evaluation suites.
