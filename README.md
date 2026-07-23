
# 💧 AI Water Quality Predictor

A beautiful Streamlit application that estimates drinking-water potability from common water-quality measurements. It includes interactive single predictions, dataset exploration, batch CSV predictions, downloadable reports, and four trained classification models.

⚠️ **Important Notice**: This project is an educational prediction tool. It does not replace laboratory testing, local regulations, or professional public-health advice.

---

## ✨ Features

- 🎯 **Manual Prediction**: Enter nine water-quality parameters to predict potability
- 🎲 **Example Inputs**: Load safe, unsafe, or random dataset samples
- 🤖 **Multiple ML Models**: Random Forest, XGBoost, Decision Tree, Logistic Regression
- 🛠️ **Smart Preprocessing**: Median imputation and standardized scaling
- 📊 **Dataset Explorer**: View summaries, distributions, and correlation heatmaps
- 📈 **Batch Prediction**: Upload a CSV for bulk predictions and download results
- 🚨 **Rule-Based Warnings**: WHO/EPA guideline alerts for pH, turbidity, chloramines, and sulfate
- 📥 **Downloadable Reports**: Save single prediction reports to text files

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or later
- Git (optional)

### Step 1: Navigate to the project
```powershell
cd "x:\water quality prediction\water-quality-prediction"
```

### Step 2: Install dependencies
```powershell
python -m venv venv

# For Windows PowerShell
.\venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Run the app
```bash
python -m streamlit run app.py
```

Open your browser and go to **http://localhost:8501** to start using the app!

---

## 🔄 Train/Retrain Models

If the `models/` folder is missing or you've updated your dataset:
```bash
python train_model.py
```

This creates the following in `models/`:
- `Random_Forest.pkl`
- `XGBoost.pkl`
- `Decision_Tree.pkl`
- `Logistic_Regression.pkl`
- `scaler.pkl`
- `imputer.pkl`
- `feature_names.pkl`

---

## 📋 Input Parameters

| # | Parameter | Description |
|---|-----------|-------------|
| 1 | ph | pH level (WHO recommended: 6.5-8.5) |
| 2 | Hardness | Calcium and magnesium concentration |
| 3 | Solids | Total dissolved solids (TDS) |
| 4 | Chloramines | Chlorine and ammonia compound (EPA limit: 4 mg/L) |
| 5 | Sulfate | Sulfate ion concentration (WHO limit: 250 mg/L) |
| 6 | Conductivity | Electrical conductivity |
| 7 | Organic_carbon | Total organic carbon |
| 8 | Trihalomethanes | Disinfection byproducts |
| 9 | Turbidity | Cloudiness of water (WHO limit: &lt;5 NTU) |

---

## 📁 Project Structure

```
water-quality-prediction/
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── dataset/
│   └── water_potability.csv        # Training dataset
├── models/                         # Saved ML models and artifacts
├── debug.py                        # Diagnostic helper
└── test_specific.py                # Single-input model checker
```

---

## 📈 Batch Prediction

Upload a CSV containing all 9 input parameters (extra columns allowed). The app adds these columns to your results:
- `Prediction`: SAFE/UNSAFE
- `Confidence`: Model confidence percentage
- `Prob_UNSAFE`: Probability of unsafe water (%)
- `Prob_SAFE`: Probability of safe water (%)

Missing values are automatically filled using the trained median imputer.

---

## 📝 Notes

Potability predictions depend on the training dataset and are probabilistic. A result labeled `SAFE` means the selected model predicts class `1` — it is not a certification that the water is safe to drink. Always follow applicable water-testing standards and seek qualified advice for real-world decisions.

---

## 📜 License

Educational use only. No warranty provided.
