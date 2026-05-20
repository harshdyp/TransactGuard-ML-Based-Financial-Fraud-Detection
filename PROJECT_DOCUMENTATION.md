# FraudX — Project Document

## 1) Outcomes of Project
- **Fraud detection system** that assigns a **Suspicion Score** \((0.0–1.0)\) to each transaction and flags suspicious records based on a configurable threshold.
- **Explainable output** for flagged transactions via **human-readable reasons** (`Suspicion_Reasons`) and a **rule-based primary label** (`Fraud_Type`).
- **End-to-end pipeline** from **synthetic data generation + model training** to **interactive analysis** in a UI.
- **Multiple access options**:
  - **Streamlit UI** for analysts (upload, tune threshold, visualize, export)
  - **Optional FastAPI** endpoints for programmatic scoring

## 2) Project Architecture
### High-level components
- **Data layer**
  - Input datasets: `transactions.csv` (generated) or user-uploaded CSVs
  - Risk lists: `known_fraudsters.csv`, `sanctioned_entities.csv`
- **Training & artifact generation**
  - `main.py`: generates synthetic transactions, trains **RandomForest**, saves artifacts
  - Artifacts saved as: `fraud_detection_model.joblib`, `fraud_detection_scaler.joblib`, `feature_importances.joblib` (and optionally `model_metrics.joblib`)
- **Detection & explanation engine**
  - `suspicious_by_model.py`:
    - `preprocess_data`: aligns uploaded data to training feature schema
    - `detect_suspicious_transactions`: computes `Suspicion_Score` using model probability, flags `Is_Suspicious` using the chosen threshold
    - `classify_fraud_type`: assigns a primary fraud label using prioritized rules
    - `explain_suspicion`: generates readable explanations (`Suspicion_Reasons`)
- **Presentation layer**
  - `app.py`: Streamlit app for upload → detection → dashboards → download/export
- **Optional service interface**
  - `api/main.py`: FastAPI service exposing health and scoring endpoints (single/batch)

### Data flow (end-to-end)
1. **User uploads CSV** (or uses generated `transactions.csv`)
2. **Preprocessing** aligns features to training schema (encoding + scaling)
3. **Model scoring** using `predict_proba` → `Suspicion_Score`
4. **Decision** via threshold slider → `Is_Suspicious`
5. **Explain + label** → `Suspicion_Reasons`, `Fraud_Type`
6. **Outputs**: dashboard visuals + export (e.g., `suspicious_transactions.csv`)

## 3) Outcomes Achieved Till Date
- **Working Streamlit UI** (`app.py`) with:
  - CSV upload
  - Threshold slider for suspicion cutoff
  - Visualizations (KPIs, distributions, fraud-type summaries, geo views where applicable)
  - Download/export of suspicious rows including `Fraud_Type` and `Suspicion_Reasons`
- **Model training pipeline** (`main.py`) producing reproducible inference artifacts (`*.joblib`).
- **Detection engine implemented** (`suspicious_by_model.py`) that:
  - preprocesses and aligns features
  - scores transactions with RandomForest probability
  - flags suspicious cases and produces explanations
  - assigns a primary fraud-type category (rule-based)
- **Documented workflow and architecture** in `README.md`.
- **Optional FastAPI layer** (`api/main.py`) available for programmatic access (if used in deployment/demo).

## 4) Current Results and Next Phase Plan
### Current results (as reported in project documentation)
- **Dataset**: 10,000+ synthetic transactions, 20+ engineered features
- **Model metrics (reported)**:
  - Accuracy: **95.2%**
  - Precision: **94.8%**
  - Recall: **93.5%**
  - F1-score: **94.1%**
- **Operational result**: each transaction receives:
  - `Suspicion_Score` (probability of fraud)
  - `Is_Suspicious` (threshold-based)
  - `Fraud_Type` (rule-based category)
  - `Suspicion_Reasons` (explainable text)

### Next phase plan
- **Evaluation upgrades**
  - Add PR-AUC / ROC-AUC, confusion matrix reporting, and threshold tuning using cost/PR curve.
  - Handle class imbalance explicitly (if switching to real data) using weighting or sampling.
- **Explainability upgrades**
  - Add SHAP-based per-transaction feature contribution explanations alongside current reason strings.
  - Store explanation metadata for auditability.
- **Data & robustness**
  - Stronger schema validation for uploaded CSVs with clear error reporting.
  - Add unit tests for preprocessing, scoring, fraud-type rules, and explanation logic.
- **Scalability & deployment**
  - Make FastAPI scoring the primary backend for production-style deployment.
  - Containerize (Docker) and add simple CI checks.
- **Product improvements**
  - Analyst workflow: case review status, feedback loop, and retraining triggers.

