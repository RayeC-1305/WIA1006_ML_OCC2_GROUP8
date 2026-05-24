# Setup Guide — Tying the (Data) Knot

## Prerequisites

- **Python 3.10+** (project was developed with Python 3.14)
- **pip** (comes with Python)
- **Git** (to clone the repository)

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/WIA1006_ML_OCC2_GROUP8.git
cd WIA1006_ML_OCC2_GROUP8
```

---

## 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\activate` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Full Dependency List

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical plots (heatmaps, boxplots) |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `xgboost` | XGBoost gradient boosting model |
| `shap` | SHAP explainability plots |
| `imbalanced-learn` | SMOTE / SMOTEENN for class imbalance |
| `streamlit` | Web dashboard frontend |
| `plotly` | Interactive charts in the dashboard |

---

## 4. Run the ML Pipeline (Console)

This trains all models, generates evaluations, and saves visualizations:

```bash
python main.py
```

**Outputs:**
- `models/` — Trained `.pkl` model files
- `reports/` — Model comparison CSV tables
- `visualizations/` — 13 high-resolution PNG plots

> Note: Training all models takes several minutes due to hyperparameter tuning.

---

## 5. Run the Web Dashboard (Frontend)

```bash
python -m streamlit run streamlit_app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

### Dashboard Sections

| Tab | Description |
|---|---|
| **Match Survey** | Answer a guided questionnaire to get a match prediction |
| **Custom Prediction** | Manually adjust all 15 features and predict outcomes |
| **Model Comparison** | View performance tables and all evaluation visualizations |
| **Data Explorer** | Interactive dataset exploration (distributions, correlations, stats) |
| **Project Report** | Full rendered group project report |

---

## 6. Run the Original Tkinter GUI (Optional)

```bash
python gui_app.py
```

> Requires the `xgboost.pkl` model to exist in `models/` (run `main.py` first).

---

## Troubleshooting

### `streamlit` command not found

Use the module form instead:

```bash
python -m streamlit run streamlit_app.py
```

### Packages not found (IDE shows import errors)

Make sure your IDE is using the `.venv` Python interpreter:

- **VS Code:** Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv\Scripts\python.exe`
- **PyCharm:** Settings → Project → Python Interpreter → add `.venv`

### Models missing / prediction errors

Run the full pipeline first to generate model files:

```bash
python main.py
```

### `DLL load failed` or numpy/scikit-learn errors

Upgrade pip and reinstall:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```
