# Water Quality Intelligence

An environmental data analytics platform for dataset-based water-potability classification. **0 = Not Potable; 1 = Potable.** Neither class is a drinking-water safety certification. Laboratory testing and applicable water-quality standards remain essential; there is no live monitoring or sensor connection.

---

## Key Features & UI/UX Capabilities

- **Modern Analytical Interface:** Built with a clean, structured design inspired by enterprise analytics products—minimalist light slate background, deep navy brand surfaces, and refined teal accent palette.
- **Light & Dark Mode Support:** Instant theme switching between Light Mode (`☀️ Light`) and Dark Mode (`🌙 Dark`) via the sidebar `APPEARANCE` toggle, with automatic Plotly chart and metric palette adaptation.
- **High-Contrast Grouped Navigation:**
  - **WORKSPACE:** System Overview, Single Sample Prediction, Batch CSV Analysis.
  - **ANALYTICS:** Dataset Quality Audit, Model Benchmark & Performance.
  - **PROJECT:** Methodology Protocol & Limitations.
- **Compact Status Bar:** Persistent top status bar displaying platform context and real-time model readiness (e.g. `● Soft Voting RF SVC active · 7/7 Pipelines ready`).
- **Responsive KPI Cards:** Dynamic typography and text-wrapping ensuring zero metric clipping across narrow and wide viewport widths.
- **Balanced 3-Column Prediction Form:** Physical profile, dissolved constituents, and chemical profile parameters organized in clean columns with reference sample loaders and instant probability charts.
- **Structured 5-Step Batch Ingestion:** Clear numbered workflow for CSV template download, schema validation, preview, batch inference execution, and export.

---

## Quickstart

Run commands from the `water-quality-prediction` directory with Python and dependencies installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Open `http://localhost:8501`.

To train models or run verification tests locally:

```bash
python train_model.py
python test_models.py
python test_training.py
python test_app.py
```

Training runs nested cross-validation (approximately 5 minutes) and replaces the seven fitted Pipelines, `feature_names.pkl`, `evaluation_results.json`, `evaluation_report.txt`, and `training_summary.json` under `models/`. It preserves the baseline evaluation reports and never modifies the original dataset.

---

## Application Structure & Views

| Category | View | Key Analytical Capabilities |
|:---|:---|:---|
| **WORKSPACE** | **Overview** | Four aligned KPI metrics (Test Accuracy, Test Macro F1, Dataset Records, Input Features), active model diagnostics, 3-stage prediction pipeline overview, class distribution donut chart, and raw CSV missing-value audit. |
| **WORKSPACE** | **Water Prediction** | Reference sample loaders (*Potable*, *Not Potable*, *Random*), 3-column input form (Physical Profile, Dissolved Constituents, Chemical Profile), missing-pH imputation toggle, high-precision threshold rule disclosure, probability distribution chart, and downloadable text report. |
| **WORKSPACE** | **Batch Analysis** | 5-step sequential ingestion pipeline: format specifications & template download, automated CSV validation, row preview, execution of fitted Pipeline, summary KPIs, probability distribution histogram, and full prediction CSV export. |
| **ANALYTICS** | **Dataset Explorer** | Overview metrics, target-class balance, missing-value breakdown bar chart, interactive feature distribution histograms by class, pairwise Pearson correlation matrix heatmap, descriptive statistics, and raw data preview. |
| **ANALYTICS** | **Model Performance** | Test accuracy, Macro F1, and weighted F1 metrics, train-validation overfit gap assessment, multi-model Macro F1 comparison chart & benchmark table, confusion matrix heatmap, per-class classification report, training vs. unseen accuracy curve, and global feature importance. |
| **PROJECT** | **Training & Methodology** | Detailed step-by-step pipeline architecture (data acquisition, zero pH convention, 80/20 stratified split, nested CV, decision threshold calibration), model zoo comparison table, observed scientific limitations, and CLI commands. |

The sidebar selects from **Random Forest**, **SVC (RBF)**, **Extra Trees**, **Soft Voting Ensemble (RF + SVC)**, **XGBoost**, **Decision Tree**, or **Logistic Regression**. The default recommended model is selected by the highest mean nested cross-validation Macro F1, not raw test accuracy.

---

## CSV Inputs & Batch Output Schema

For batch prediction, upload a comma-separated UTF-8 CSV containing the nine required parameters:

```text
ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
```

- Values must be finite numbers or blank cells.
- Blank cells and `ph == 0` are handled in-memory by the fitted median imputer; original files are not modified.
- Extra columns (e.g., sample IDs, site codes) are preserved in the export.
- `Potability` is optional and is never used as an input.

The exported `batch_predictions.csv` includes all uploaded rows alongside:

| Column | Type | Meaning |
|:---|:---|:---|
| `Prediction` | String | `Not Potable` (Class 0) or `Potable` (Class 1) |
| `Predicted_Class` | Integer | Binary class identifier (`0` or `1`) |
| `Max_Class_Probability` | Float | Highest model confidence score, **0–100 (%)** |
| `Prob_Not_Potable` | Float | Probability assigned to Not Potable, **0–100 (%)** |
| `Prob_Potable` | Float | Probability assigned to Potable, **0–100 (%)** |
| `Prediction_Model` | String | Identifier of the fitted pipeline used for inference |

---

## Machine Learning Methodology

1. **Dataset & Partitioning:**
   - 3,276 records from `dataset/water_potability.csv` across nine numeric parameters and binary target `Potability` (39% Potable / 61% Not Potable).
   - Stratified **80% training / 20% test** split (`random_state=42`). The test partition is evaluated strictly once after all tuning and thresholds are frozen.
2. **Leakage-Free Preprocessing:**
   - Every candidate model is structured as a scikit-learn `Pipeline`:
     $$\text{SimpleImputer}(\text{strategy}=\text{"median"}) \longrightarrow \text{StandardScaler}() \longrightarrow \text{Classifier}$$
   - Imputation and scaling statistics are computed exclusively on training folds.
3. **5×5 Nested Cross-Validation:**
   - Hyperparameters for Random Forest, SVC, and Extra Trees are tuned across 5 inner folds; outer folds benchmark generalization without data leakage.
   - Paired corrected t-tests (Nadeau–Bengio) verify whether score differences exceed statistical noise.
4. **Decision Threshold Calibration:**
   - Because false Potable classifications carry health risks, decision thresholds are calibrated on out-of-fold training predictions to achieve Potable precision $\ge 0.70$ where feasible.
5. **Baseline Benchmark:**
   - All models are benchmarked against a majority-class heuristic baseline (~61.0% accuracy).

---

## Scientific Limitations & Boundary Conditions

- **Overfitting & Generalization:** Ensemble models demonstrate high fold-training accuracy (~85–90%) but hold-out test accuracy of ~67–68%, indicating significant generalization limits.
- **Missing Chemical & Biological Parameters:** The dataset evaluates only nine physical/chemical features. It excludes microbiological pathogens (e.g., *E. coli*, coliform bacteria), toxic heavy metals (lead, arsenic, mercury), PFAS, microplastics, and radiological contaminants.
- **Statistical Margin of Error:** Differences between top-performing ensemble models fall within one cross-validation standard deviation.
- **Threshold Trade-offs:** Prioritizing high precision for Potable classifications reduces false positives at the expense of lower Potable recall.
- **Educational Prototype:** This platform is an educational machine learning demonstrator. It must never replace certified laboratory testing or drinking-water regulatory compliance evaluations.
