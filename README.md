# 🧠 Parkinson’s Disease Classifier (XGBoost)

This project builds a compact, high-performance classifier to detect Parkinson’s disease using biomedical voice features from the [UCI Parkinson’s dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons).

The final model uses only the top 6 most important features and achieves over **92% accuracy** while maintaining high precision and recall — even with class imbalance.

---

## ✅ Project Highlights

- **Model**: Tuned XGBoost Classifier (with baseline and untuned comparisons)
- **Feature Set**: Top 6 most important voice features (feature selection based on XGBoost importance scores)
- **Imbalance Handling**: `scale_pos_weight` based on class ratio
- **Model Format**: Saved using `pickle` for portability and reuse

---

## 🔝 Top 6 Most Important Features

| Rank | Feature     | Description |
|------|-------------|-------------|
| 1️⃣   | `mdvp:fhi`  | **Maximum fundamental frequency (Hz)** — Highest vocal pitch. Parkinson’s can cause instability, increasing max frequency irregularities. |
| 2️⃣   | `mdvp:fo`   | **Average fundamental frequency (Hz)** — Average vocal pitch. Often reduced or unstable in Parkinson’s speech. |
| 3️⃣   | `spread1`   | **Nonlinear signal spread** — Measures asymmetry in the voice signal; higher values may reflect vocal tremor or breathiness. |
| 4️⃣   | `rpde`      | **Recurrence Period Density Entropy** — Quantifies unpredictability in the signal. Elevated in disordered voices. |
| 5️⃣   | `d2`        | **Correlation dimension** — Measures complexity of the vocal system. Lower complexity often seen in Parkinson’s. |
| 6️⃣   | `spread2`   | **Second nonlinear spread measure** — Complements `spread1`; also tracks signal deviation and dysphonia symptoms. |

---

### 🧠 Takeaway

The top 6 features identified by XGBoost — primarily nonlinear vocal measures like `mdvp:fhi`, `rpde`, and `spread1` — capture **instability**, **irregularity**, and **reduced complexity** in voice signals. These characteristics are strongly associated with **Parkinson’s-induced dysphonia**, making them highly predictive even in a compact model.

---

## 📊 Final Model Performance

| Metric        | Class 0 (Healthy) | Class 1 (Parkinson’s) | Macro Avg | Weighted Avg |
|---------------|-------------------|------------------------|-----------|---------------|
| **Precision** | 0.89              | 0.93                   | 0.91      | 0.92          |
| **Recall**    | 0.80              | 0.97                   | 0.88      | 0.92          |
| **F1-score**  | 0.84              | 0.95                   | 0.90      | 0.92          |
| **Support**   | 10                | 29                     | –         | 39            |
| **Accuracy**  | –                 | –                      | –         | **92.3%**     |

---

## 🔢 Confusion Matrix

| Actual \ Predicted | 0 (Healthy) | 1 (Parkinson’s) |
|--------------------|-------------|------------------|
| **0 (Healthy)**    | 8           | 2                |
| **1 (Parkinson’s)**| 1           | 28               |

---

## 📂 Saved Models

| File Name                        | Description                                                         |
|----------------------------------|----------------------------------------------------------------------|
| `parkinsons_xgb_top6_tuned.pkl` | 🔧 Tuned XGBoost model using top 6 features (RandomizedSearchCV object) |
| `xgboost_baseline_model.pkl`    | ⚙️ Baseline XGBoost model with default hyperparameters              |
| `parkinsons_xgb_top6.pkl`       | 🧪 Untuned XGBoost model using top 6 selected features               |

---

## 🔧 Requirements

All dependencies are listed in `requirements.txt`. Key packages include:

| Package         | Purpose                                |
|-----------------|----------------------------------------|
| `xgboost`       | Gradient-boosted trees for classification |
| `scikit-learn`  | Model evaluation, cross-validation     |
| `ucimlrepo`     | Load datasets from the UCI ML repository |
| `pandas`        | Data handling and preprocessing        |
| `numpy`         | Numerical operations                   |
| `matplotlib`    | Data visualization                     |
| `seaborn`       | Statistical visualizations             |

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/parkinsons-xgb.git
cd parkinsons-xgb
pip install -r requirements.txt

import pickle

# Load top 6 feature list
with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

# Load tuned model (RandomizedSearchCV object)
with open('parkinsons_xgb_top6_tuned.pkl', 'rb') as f:
    rs = pickle.load(f)
best_model = rs.best_estimator_

# Load baseline model
with open('xgboost_baseline_model.pkl', 'rb') as f:
    baseline_model = pickle.load(f)

# Load untuned top-6 model
with open('parkinsons_xgb_top6.pkl', 'rb') as f:
    top6_model = pickle.load(f)

# Example prediction
# X_new is a pandas DataFrame with the same structure as training features
y_pred = best_model.predict(X_new[top_features])
