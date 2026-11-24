# 🤖 Agentic AI Data Scientist (v2.0 - Production Standard)

An enterprise-grade **Agentic AI Data Scientist** that autonomously investigates datasets, diagnoses data health, builds leak-free preprocessing pipelines, benchmarks machine learning algorithms with cross-validation, explains feature importance, generates executive reports, and exports deployment-ready models.

---

## 🌟 Why This Architecture? (Real-World vs. Toys)

| Capability | Basic Prototype (Legacy) | **Agentic AI Data Scientist v2.0** |
|---|---|---|
| **Execution Paradigm** | Hardcoded script without reasoning | **Autonomous Agent with Thought-Action-Observation loops** |
| **Data Memory** | Reads CSV repeatedly from disk, discards state | **Stateful in-memory pipeline** across all lifecycle phases |
| **Preprocessing** | `df.dropna()` (drops all rows if nulls exist) | **Median/Mode Imputation, Variance Thresholding & Scalers** |
| **Leakage Protection** | `pd.get_dummies()` before train/test split | **`ColumnTransformer` fitted strictly on train split** |
| **Model Validation** | Single 80/20 train/test split | **5-Fold Stratified / K-Fold Cross-Validation + Holdout Set** |
| **Algorithms** | Basic Logistic/Linear Regression | **Logistic/Ridge, Random Forest & HistGradientBoosting (LightGBM-grade)** |
| **Explainability** | None | **Top Feature Importance analysis & Holdout Confusion Matrices** |
| **Artifacts & Export** | None | **Self-contained `.joblib` model bundle & standalone Python inference code** |
| **Reporting** | Raw console dump | **Comprehensive Executive Briefing (Markdown & HTML)** |
| **LLM Flexibility** | Hardcoded local Ollama | **Official Google GenAI (`gemini-2.5-flash`), OpenAI, or Built-in Specialist** |

---

## 📁 Project Structure

```text
agentic-ai-data-scientist/
├── agent/
│   ├── orchestrator.py     # Autonomous agent coordinator
│   └── llm_client.py       # Multi-provider client (Gemini, OpenAI, Heuristic fallback)
├── core/
│   └── state.py            # Typed state container for the investigation
├── engine/
│   ├── profiler.py         # Statistical profiling, health score & correlation
│   ├── cleaner.py          # Leak-free ColumnTransformer preprocessing
│   ├── modeler.py          # Multi-algorithm CV benchmarking & serialization
│   └── reporter.py         # Publication-ready executive report generator
├── data/
│   └── sample_churn.csv    # Real-world benchmark dataset for one-click testing
├── outputs/
│   └── models/             # Exported .joblib bundles
├── app.py                  # Full-featured Streamlit interactive studio
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Activate the Virtual Environment

```bash
# On Windows PowerShell:
C:\Users\KIIT\.gemini\antigravity\scratch\venv\Scripts\Activate.ps1
```

### 2. Run the Application

```bash
streamlit run app.py
```

---

## 🧠 LLM Providers Supported

1. **Built-in Heuristic Specialist (Default - No Setup Required)**:
   - Works 100% offline without any API keys or network dependencies.
2. **Google Gemini**:
   - Provide your `GEMINI_API_KEY` in the sidebar or set it as an environment variable (`$env:GEMINI_API_KEY="your-key"`).
   - Powered by the official Google GenAI SDK with models like `gemini-2.5-flash`.
3. **OpenAI**:
   - Provide your `OPENAI_API_KEY` in the sidebar or via environment variable.

---

## 📊 End-to-End Workflow

1. **Upload Dataset:** Upload any tabular CSV or click **"Load Telecom Churn Dataset"** in the sidebar.
2. **Select Target Column:** Choose the column to predict. The agent automatically infers Binary Classification, Multiclass Classification, or Regression.
3. **Run Autonomous Agent:**
   - **Step 1:** Ingests data, diagnoses quality, and computes a **Data Health Score (0-100)**.
   - **Step 2:** Drops zero-variance & high-cardinality IDs, imputes missing values, and builds an isolated `ColumnTransformer`.
   - **Step 3:** Benchmarks candidate algorithms with 5-Fold Cross-Validation and evaluates holdout metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, R²).
   - **Step 4:** Exports the winning model pipeline to `outputs/models/best_pipeline.joblib`.
   - **Step 5:** Compiles a publication-ready **Executive Investigation Report**.
4. **Inspect & Deploy:**
   - Review agent thoughts and reasoning traces.
   - Explore interactive Plotly correlation heatmaps, feature importance charts, and confusion matrices.
   - Test predictions on single records directly in the browser and download the `.joblib` bundle.
