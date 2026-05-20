# Syngenta-style interview prep — FraudX

Supporting material for explaining [FraudX](README.md) in technical and regulated-enterprise conversations. Aligns with code in `app.py`, `suspicious_by_model.py`, `main.py`, and optional `api/main.py`.

---

## 1) Narratives

### About two minutes (elevator pitch)

FraudX is an analyst-facing fraud triage prototype. It scores each row of transaction data with a supervised model (Random Forest) trained on synthetic—but structurally realistic—payments data with about twenty engineered signals: geography, payment rail, velocity, sanction and known-fraudster flags, device and anonymity cues. The Streamlit UI lets someone upload a CSV, set a suspicion threshold on the model’s predicted fraud probability, and get back per-row outputs: `Suspicion_Score`, `Is_Suspicious`, a prioritized rule-based `Fraud_Type`, and plain-language `Suspicion_Reasons` driven by global feature importance. Results can be explored with charts and exported. The pipeline is intentionally simple to demonstrate end-to-end ML plus explainability; there is also an optional FastAPI layer for programmatic scoring if you treat the model as a service.

### About ten minutes (deep walk-through)

**Problem and user.** Payment and compliance teams review large volumes of events. FraudX narrows focus: rank transactions by risk and attach a coarse label plus reasons so an analyst can start from the highest-impact cases instead of scrolling raw tables.

**Data and training.** `main.py` generates on the order of ten thousand labeled transactions (~5% fraud in the generator), persists risk lists (`known_fraudsters.csv`, `sanctioned_entities.csv`), fits a Random Forest plus `StandardScaler` on one-hot-encoded categoricals aligned to a fixed feature schema, and saves `fraud_detection_model.joblib`, `fraud_detection_scaler.joblib`, `feature_importances.joblib`, and optionally `model_metrics.joblib`. That makes training reproducible and keeps inference tied to a known column order via `scaler.feature_names_in_`.

**Inference and alignment.** Uploaded files may omit columns or use aliases; the UI can map columns, and `suspicious_by_model._ensure_required_columns` fills gaps with conservative defaults before `preprocess_data` runs `get_dummies` and zero-fills unseen training columns so the scaler always sees the same width as at fit time.

**Scoring.** `predict_proba` for the fraud class becomes `Suspicion_Score`; the UI threshold turns that into `Is_Suspicious`. A separate rule layer assigns exactly one primary `Fraud_Type` (known fraudster, sanctioned entity, cross-border patterns, device evasion, ATO cues, brute force, odd-hour behavior, etc.) following a documented priority order in the README so labels stay consistent.

**Explainability.** Explanations are heuristic: compare important features against sender history where useful, and attach short text snippets. That is explainable enough for demos; it is not a substitute for audited SHAP or full model-cards.

**Presentation.** `app.py` is Streamlit: KPIs, threshold sensitivity, distributions, geography views where data supports them, downloads of suspicious subsets. Deployment today is demonstration-grade; batch or real-time APIs would emphasize the existing FastAPI sketch and hardened auth, lineage, and monitoring.

**Reported metrics (synthetic validation).** Documentation cites roughly 95%+ accuracy-style metrics on the synthetic split—useful only to show the baseline model is learning structure, not as a production guarantee.

---

## 2) Talking points — documented gaps and “next phase” themes

These mirror [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) and are safe to cite as intentional future work if interviewers probe limitations.

### Synthetic data and generalization

- Training data is generated with explicit fraud prevalence and heightened risk correlations for fraudulent rows (`main.py`). That validates the pipeline mechanics but **overstates** typical real-world realism: label definitions, adversarial adaptation, seasonality, and rare fraud classes do not naturally appear.
- **Interview line:** “I’d treat current metrics as a plumbing check; promotion would require historical production labels, leakage review, and time-based splits.”

### Class imbalance and evaluation

- Fraud is sparse in production; **accuracy** can mask poor recall at the rare class. Mention **precision–recall curve**, **PR-AUC**, **cost-sensitive thresholding**, and stakeholder-owned **expected cost** per false negative vs false positive.
- Stratified or **rolling-origin** evaluation prevents optimistic scores when fraud patterns drift over time.

### Model drift and data drift

- **Schema drift:** new categorical values become new one-hot columns that are absent at inference and zero-filled—which is coherent but statistically changes meaning; better handling includes hashing, embeddings, explicit “unknown” buckets, or retraining pipelines.
- **Score drift:** monitor distribution of `Suspicion_Score`, flag volume, precision on sampled audits, and back-test rules—same ideas apply to rebate or channel abuse.

### Explainability beyond feature-importance snippets

- Global importances plus rule-based reasons are **summaries**, not causal proofs. SHAP or similar per-row attribution, plus immutable **explainability payloads** stored with decisions, strengthens audit defenses.
- Be clear what you claim: narrative alignment with contributing factors versus mathematically attributable contributions.

### Robustness gaps called out in project docs

- Stronger CSV **schema validation** and user-facing validation errors instead of silent defaults for missing columns.
- **Unit tests** for preprocessing, rule priority, explanations, and API contracts.
- **Containerization**, CI, and making FastAPI the primary inference path behind auth and quotas for production posture.

---

## 3) Syngenta / agritech scenario bridges (concrete analogies)

Use these if interviewers pivot from banking-style transactions to crop protection or seed-channel risk.

### 1. Sender / receiver ↔ channel counterparties

**Transaction:** payer and payee with countries and rails.  
**Channel analogy:** rebate or incentive claim flows from grower ↔ distributor ↔ retailer ↔ Syngenta. “Sender velocity” aligns with abnormal claim submission rate per distributor SKU or per territory; cross-border parallels cross-regional shipment versus registered sales footprint.

### 2. Sanctions and known fraudster lists ↔ denial lists

**Implementation:** IDs matched against CSV-backed risk lists in the synthetic generator and feature flags (`Is_Known_Fraudster`, `Is_Sanctioned_Entity`).  
**Interview mapping:** analogous to **blocked counterparty lists**, counterfeit hot spots, distributors under suspension, or parties associated with diverted product—all require fresh list governance and lawful basis.

### 3. VPN / IP / new device ↔ physical and digital channel inconsistency

**Transaction:** anonymity and device fingerprints.  
**Analogy:** **ship-to vs bill-to discrepancies**, impossible routing for seed lots, barcode or redemption patterns inconsistent with agronomic zoning, sudden distributor behavior unlike its peer cohort—all are “signals that don’t match history or geography.”

### 4. Velocity and amount ↔ rebate abuse and phantom volume

High amount plus high velocity in FraudX parallels **burst claims**, **duplicate submissions**, or **stacked rebates** inconsistent with seeded acres or shipments in ERP—not seasonality alone, but deviation from distributor baselines adjusted for planting window.

### 5. Seasonality and geography ↔ territory-compliant models

Peak legitimate activity at planting parallels payment spikes—but models should ingest **normalized baselines**: rolling z-scores by region and crop week, incorporation of agronomic calendars, or hierarchical models by sales region so “busy season” isn’t punished as suspicious without context.

---

*End of prep notes.*
