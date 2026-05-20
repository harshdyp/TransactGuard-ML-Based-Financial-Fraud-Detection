# Answers: Syngenta-style questions on FraudX

Grounded in this repo: Streamlit [`app.py`](app.py), scoring and rules [`suspicious_by_model.py`](suspicious_by_model.py), training [`main.py`](main.py), optional [`api/main.py`](api/main.py).

---

## 1) Project walkthrough

### What problem does FraudX solve, and who is the user?

**Answer:** FraudX helps **risk or fraud analysts triage transactions** faster. Each row gets a probabilistic **`Suspicion_Score`** and a binary **`Is_Suspicious`** flag (from an adjustable threshold). It adds **routing context** (`Fraud_Type`) and **short rationales** (`Suspicion_Reasons`) so a human can review the highest-priority cases and export results. The primary user in this repo is the **analyst operating the Streamlit dashboard**, not an unattended real-time payment switch (though a service-style API is sketched separately).

### Walk through the end-to-end flow.

**Answer:**

1. **Input:** CSV upload (or bundled demo data) with transaction-level fields; optional column mapping in the UI.
2. **Schema safety:** [`_ensure_required_columns`](suspicious_by_model.py) fills missing expected columns with defaults so scoring does not crash.
3. **Preprocess:** [`preprocess_data`](suspicious_by_model.py) one-hot encodes `Sender_Country`, `Receiver_Country`, `Payment_Method`, `Transaction_Currency`, adds any training-time columns missing from the batch as zeros, and reorders columns to match `scaler.feature_names_in_`.
4. **Scale & score:** `StandardScaler.transform` then `RandomForestClassifier.predict_proba`; class-1 probability is stored as **`Suspicion_Score`**.
5. **Decision:** `Suspicion_Score > threshold` → **`Is_Suspicious`** (threshold is user-controlled in the UI).
6. **Rules layer:** [`classify_fraud_type`](suspicious_by_model.py) assigns a **single primary** `Fraud_Type` using a **fixed priority** list (see section 4).
7. **Explain:** [`explain_suspicion`](suspicious_by_model.py) builds text from **global feature importances** plus simple comparisons to the sender’s history (e.g. amount/velocity vs average), with fallbacks based on score bands.
8. **Output:** Dashboards and **download** of flagged rows with scores, types, and reasons.

### Why Random Forest vs logistic regression, gradient boosting, or anomaly detection?

**Answer:** In [`main.py`](main.py) a **Random Forest** is used (`n_estimators=100`, depth and `min_samples_split` capped). **Pros for this project:** handles mixed types after encoding, captures **non-linear** interactions and thresholds without hand-tuning many basis functions, gives **feature importances** that the explanation path reuses, and is robust on tabular data of moderate size. **Logistic regression** would be a stronger baseline for linearly separable signals and easier regulatory narrative, but may underfit cross-feature effects. **Gradient boosting** (e.g. XGBoost, LightGBM) often wins on leaderboard-style tabular fraud with tuning and proper cross-validation. **Anomaly detection** is useful when **labels are scarce** or fraud is non-stationary; here the pipeline is **supervised** on `Is_Fraudulent` from synthetic data, so RF matches the chosen problem framing. In production you would compare calibrated models (including boosting) using **business-weighted** metrics, not accuracy alone.

### What is synthetic data doing, and what breaks in production?

**Answer:** [`main.py`](main.py) generates ~10k rows with **`FRAUD_PERCENTAGE = 0.05`**, exaggerates risky patterns for fraudulent rows (e.g. higher velocity, VPN, IP change), and writes `transactions.csv` plus list files. Synthetic data **proves the pipeline** (train → persist joblib artifacts → score in UI) and avoids sharing real PII.

**What breaks or weakens at production scale:**

- **Distribution shift:** real fraud evolves; training joint distribution no longer matches live traffic.
- **Label quality:** production labels are delayed, biased by investigation capacity, or inconsistent.
- **Imbalance:** real fraud rates are often far below 5%; accuracy can look good while recall at fixed precision is poor.
- **Feature availability:** ERP/PSP feeds differ from the generator; missing or late features change the effective model.
- **Train/serve skew:** training in `main.py` applies `get_dummies` to **more** categorical columns than inference’s [`preprocess_data`](suspicious_by_model.py) encodes explicitly; uploads that omit those columns rely on **zero-filled** dummy columns for the extra training features—manageable for demos but **must be unified** for a serious deployment (one shared preprocessing module, versioned with the model).

---

## 2) Machine learning and evaluation

### How do you measure success (accuracy, precision, recall, F1)? Which metric matters most for fraud?

**Answer:** [`main.py`](main.py) reports **accuracy, precision, recall, F1**, and a confusion matrix; the Streamlit sidebar shows similar headline numbers from `model_metrics.joblib` or defaults in [`app.py`](app.py). For fraud, **no single metric is always “the one”**—the business pays for **false negatives** (missed fraud) and **false positives** (analyst load, customer friction) differently. **Precision** answers “of what we flag, how much is real?” **Recall** answers “of all fraud, how much did we catch?” Typically teams care about **recall at a minimum acceptable precision** (or **precision at a fixed review capacity**). **Accuracy** is misleading when fraud is rare.

### How would you validate on imbalanced real fraud (PR-AUC, cost curves, stratified validation)?

**Answer:**

- Use **PR-AUC** (average precision) and **ROC-AUC** together; for rare positives, **PR curves** are often more informative.
- **Stratified k-fold** or **time-based splits** (no random shuffle across time) to avoid leakage and optimistic scores.
- **Cost-sensitive thresholding:** pick threshold by minimizing expected cost \(C_{FP} \cdot FP + C_{FN} \cdot FN\) or by constraining FP rate.
- **Calibration** (Platt scaling, isotonic) if scores are used as “probabilities” for policy.
- **Backtesting** on frozen historical periods before each release.

### How does the threshold trade off precision vs recall? Who owns it?

**Answer:** Raising the threshold **flags fewer** rows → usually **higher precision, lower recall**; lowering it does the opposite. In FraudX the slider is an explicit product choice: **operations or the fraud policy owner** should own the threshold against SLAs and cost; data science provides **curves and scenario tables**, not a single technical optimum without business input.

### Feature importance vs SHAP—what does your explainability prove?

**Answer:** FraudX uses **global** importances from the forest (saved in `feature_importances.joblib`) and **rule-based** text in [`explain_suspicion`](suspicious_by_model.py). That shows **which factors the pipeline emphasizes** and gives analysts **hooks** for review; it does **not** prove causality or give a **consistent local attribution** for every row. **SHAP** (or Treeshap for tree models) approximates additive contributions **per prediction**—stronger for “why this transaction?” audits. Honest framing: current explanations are **heuristic plus global importance**, useful for demos; regulated production often needs **versioned explanations** stored with each decision.

---

## 3) Data, features, and drift

### How does `preprocess_data` align uploads to training? New countries/currencies never seen?

**Answer:** Fresh categoricals produce **new dummy columns** under `pandas.get_dummies` for the four encoded columns. The scaler expects **exactly** `scaler.feature_names_in_`; any training-time column absent after encoding is **added as zeros** [`preprocess_data`](suspicious_by_model.py). So a **new country** yields a **new** `Sender_Country_X` column in the batch frame—but if that column was **not** in the scaler’s vocabulary, it is **not** selected in the reorder step: only columns in `feature_names_in_` are kept. **Effect:** novel categories are mostly expressed through **whatever training-time dummies overlap** plus numeric/boolean inputs; unseen levels do not automatically add new weight unless you retrain and expand the schema. Production systems often use **explicit “unknown”** buckets, **hashing**, or **embedding** layers to handle this cleanly.

### Cold start: `_ensure_required_columns`—what is the risk?

**Answer:** Missing fields become **neutral or “safe” defaults** (`Unknown`, `UNK`, zeros, false flags)—see [`REQUIRED_INPUT_COLUMNS`](suspicious_by_model.py). **Risk:** the model may **score too low** (silent under-risk) or the defaults may correlate spuriously with training. **Mitigation:** strict validation UX, refusal to score incomplete critical fields, explicit “missing” indicators as features, and logging missingness rates.

### How detect model drift or concept drift?

**Answer:**

- **Score drift:** PSI or KS tests on `Suspicion_Score` and key inputs vs training baseline.
- **Label drift:** delayed feedback—track precision on investigated samples, **recall proxies** where labels exist.
- **Schema drift:** alerts on new columns, unexpected categoricals, sudden missingness spikes.
- **Concept drift:** rising FP rate at fixed threshold, degradation in **stability plots** by segment (corridor, product). React with **rollback, recalibration, or retraining** with champion/challenger gates.

---

## 4) Rules + model (hybrid system)

### Why both rules and model? What if they disagree?

**Answer:** **Model** learns broad patterns from many weak signals; **rules** encode **non-negotiable policy** (e.g. sanctions, known bad actors) and **interpretable** storylines for reporting. In this code, **rules win for labeling** when they fire first: [`classify_fraud_type`](suspicious_by_model.py) checks `Is_Known_Fraudster` and `Is_Sanctioned_Entity` **before** score-only labels. **Disagreement examples:** model score **low** but list hit → type is still **Known Fraudster / Sanctioned Entity**; model **high** but no rule match → **High-Risk / Medium-Risk (Model)**. **Operational note:** `Fraud_Type` is computed for **every** row independent of whether you would filter by `Is_Suspicious` in the UI—the rules use raw flags and score, so **alignment with “only suspicious” queues** is a product decision you might tighten in code.

### Walk through one prioritized rule chain.

**Answer:** Illustrative path from [`classify_fraud_type`](suspicious_by_model.py):

1. If **`Is_Known_Fraudster`** → **Known Fraudster** (stops).
2. Else if **`Is_Sanctioned_Entity`** → **Sanctioned Entity** (stops).
3. Else if **crypto** + **cross-border** + (**high amount** ≥ 10k **or** **multi FX**) → **Crypto Cross-border Laundering**.
4. Else if **cross-border** + **multi FX** → **Cross-border FX Risk**.
5. Else if **high amount** + **high velocity** (≥ 8) → **Rapid Large Transfers**.
6. … continues through VPN/IP, new device/location, failed attempts, unusual time.
7. Else if **score ≥ max(0.8, threshold)** → **High-Risk (Model)**; elif **score ≥ threshold** → **Medium-Risk (Model)**; else **Low-Risk**.

---

## 5) Operations, security, compliance

### PII / GDPR—how would you handle retention and purpose limitation?

**Answer:** Even demo data can include **identifiers and geo**. For Syngenta-scale processing: **lawful basis** documented, **data minimization** (hash or tokenize IDs where possible), **retention schedules** per use case, **DPA** with processors, **cross-border** transfer mechanisms, **subject rights** workflow, and **privacy by design** in logs (no raw PII in application logs by default). Separate **model training** sets from **live scoring** with clear purposes.

### Auditability—reproduce a flag six months later?

**Answer:** Today you would need **artifact versioning**: exact **model version**, **scaler**, **feature list**, **code hash** of `preprocess_data` and `classify_fraud_type`, **threshold**, and **input row** (or canonical hash). Store **immutable decision records** (inputs snapshot + outputs + library versions). Without that, reproduction is weak.

### Security: CSV upload in Streamlit—production hardening?

**Answer:** Internal demos are different from **internet-facing** services. Hardening: **authn/z** (SSO, RBAC), **network** restrictions, **virus scan** and size limits on uploads, **secrets** in vaults not repos, **strip `api/main.py`-style open CORS** for production, **rate limits**, **input validation**, and **TLS**. Prefer **batch scoring** via secure pipelines over ad-hoc uploads for regulated data.

**Important:** [`api/main.py`](api/main.py) builds a **hand-rolled feature vector** that is **not** the same as [`suspicious_by_model.preprocess_data`](suspicious_by_model.py) + saved scaler. For consistent scores, production should **call one shared scoring path**.

### Rebates / gray market / counterfeit—how would the pipeline change?

**Answer:** Fraud would be tied to **contract rules**, **SKU/seed lot**, **territory**, **pricing**, **redemption timelines**, **ERP postings**. You would redefine **features** (eligibility mismatches, duplicate claims, impossible logistics) and **labels** from investigations; keep **deterministic compliance rules** layered with **learned ranking**; often **much lower base rate**, so metrics and thresholds must change.

---

## 6) Scaling and deployment

### When move from Streamlit to FastAPI plus a job queue?

**Answer:** When you need **service SLAs**, **authenticated clients**, **orchestration** (cron, airflow), **horizontal scale**, strong **audit** APIs, or **embedding** scoring inside other workflows. Streamlit stays useful for exploration; **API + queue** handles heavy CSVs and backfills without blocking interactive sessions.

### Latency SLAs vs batch nightly?

**Answer:** **Sync API** with tight p99 latency for **low-dimensional single-row** scoring after preprocessing; **batch jobs** for **millions** of rows, **complex joins**, or **expensive enrichments**. Nightly reruns recombine **offline features** that are not fresh enough for online paths; conflict resolution between online and offline scores should be defined.

---

## 7) Agribusiness bridge (scenarios)

### Growers / distributors / retailers—map to Sender/Receiver? Velocity? Sanctions?

**Answer:** Model ** Sender/Receiver ** as counterparties on a commercial event (rebate payer/receiver, ship-from/claimant, distributor vs retail endpoint). **Velocity** parallels abnormal **claim frequency** or **volume vs rolling baseline** per counterparty and SKU. **Sanctions / denylists** map to **suspended parties**, **counterfeit hotspots**, or **unauthorized traders**—with legal review and list governance.

### Seasonality—avoid flagging legitimate peaks?

**Answer:** Normalize by **season**, **region**, and **crop cycle** (e.g. z-scores within cohort-week); use **hierarchical** or **segmented** models; separate **level** (expected peak) from **shape anomalies** (wrong pattern for that peak).

### Geography vs registrations and territories?

**Answer:** Encode **allowed sales corridors**, **registration** of crop protection products, and **ship-to vs grower county**; cross-territory patterns can be features or **hard rules** where regulation demands.

---

## 8) Behavioral / leadership

### Trade-off you made?

**Answer (example framing):** Chose **fast analyst value** (threshold + downloadable suspicious subset + simple reasons) over **full MLOps** (automated retraining, SHAP, strict schema enforcement). That matches a **portfolio demo** but is explicitly **not** production-complete—documented gaps are the honest trade-off.

### Explainability vs accuracy?

**Answer:** Implemented **human-readable reasons** tied to importance and coarse **Fraud_Type** buckets, accepting that the strongest pure accuracy might come from a **black-box** ensemble with richer features. Next step would be **SHAP** plus **offline evaluation** proving the explanation aligns with calibrated risk.

### Stakeholders want to “catch everything”?

**Answer:** Translate into **capacity**: every extra alert consumes **investigation hours**. Show **ROC/PR curves**, **precision at top-k**, and **dollar-weighted scenarios**. Agree an **operating point** tied to budget and downstream losses, then **monitor** and **adjust** quarterly.

---

## Bottom-line speaking points

- FraudX demonstrates **supervised scoring + threshold + hybrid rules + narrative explanations** on **synthetic** data.
- **Production readiness** hinges on **real labels**, **better metrics**, **aligned preprocess** between train and serve, **audit/versioning**, and **privacy/security** controls—not on headline accuracy alone.
