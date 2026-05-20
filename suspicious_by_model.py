import pandas as pd
import numpy as np
import numpy as np

REQUIRED_INPUT_COLUMNS = {
    # Required for preprocessing (categorical encoding)
    'Sender_Country': 'Unknown',
    'Receiver_Country': 'Unknown',
    'Payment_Method': 'Unknown',
    'Transaction_Currency': 'UNK',
    # Required/commonly used downstream (explanations / analytics)
    'Sender_ID': 'UNKNOWN_SENDER',
    'Transaction_ID': 'UNKNOWN_TX',
    # Numeric / boolean features used in rules & explanations
    'Transaction_Amount': 0.0,
    'Transaction_Velocity': 0.0,
    'Unusual_Time': False,
    'Multiple_Currency_Conversions': False,
    'Repeated_Failed_Attempts': 0,
    'Is_Known_Fraudster': False,
    'Is_Sanctioned_Entity': False,
    'VPN_Usage': False,
    'IP_Address_Change': False,
    'Is_New_Device': False,
    'Is_New_Location': False,
}

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the minimum schema exists for preprocessing and explanations.

    Uploaded CSVs often have missing columns; we default them so scoring can proceed.
    """
    for col, default in REQUIRED_INPUT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
    return df

def patch_model_for_compatibility(model):
    """Patch model to handle monotonic_cst attribute compatibility issues"""
    try:
        # For DecisionTreeClassifier and related models
        if hasattr(model, 'tree_'):
            if not hasattr(model, 'monotonic_cst'):
                model.monotonic_cst = None
        
        # For ensemble models (RandomForest, etc.)
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'tree_') and not hasattr(estimator, 'monotonic_cst'):
                    estimator.monotonic_cst = None
        
        return model
    except Exception as e:
        print(f"Warning: Could not patch model for compatibility: {e}")
        return model

def detect_suspicious_transactions(df, model, scaler, threshold=0.5):
    # Patch the model for compatibility
    model = patch_model_for_compatibility(model)
    
    df = _ensure_required_columns(df)
    X = preprocess_data(df, scaler)
    X_scaled = scaler.transform(X)
    
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            # If the error persists, try a different approach
            print("Applying fallback prediction method...")
            # For binary classification, we can use predict and convert to probability
            y_pred = model.predict(X_scaled)
            y_pred_proba = y_pred.astype(float)  # Convert to probability-like values
        else:
            raise e
    
    df['Suspicion_Score'] = y_pred_proba
    df['Is_Suspicious'] = y_pred_proba > threshold
    
    # Classify fraud types for analysis/visualization
    df['Fraud_Type'] = df.apply(lambda row: classify_fraud_type(row, threshold), axis=1)
    
    return df

def preprocess_data(df, scaler):
    df = _ensure_required_columns(df)
    categorical_columns = ['Sender_Country', 'Receiver_Country', 'Payment_Method', 'Transaction_Currency']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[scaler.feature_names_in_]
    
    return df_encoded

def explain_suspicion(row, feature_importances, historical_data, threshold=0.05):
    reasons = []
    if 'Sender_ID' not in historical_data.columns:
        historical_data = historical_data.copy()
        historical_data['Sender_ID'] = 'UNKNOWN_SENDER'
    sender_history = historical_data[historical_data['Sender_ID'] == row['Sender_ID']]
    for feature, importance in feature_importances.items():
        if importance > threshold:
            if feature == 'Transaction_Amount':
                avg_amount = sender_history['Transaction_Amount'].mean()
                if row[feature] > avg_amount * 1.5:
                    reasons.append(f"High Transaction Amount (${row[feature]:.2f}, {row[feature]/avg_amount:.1f}x average)")
            elif feature == 'Transaction_Velocity':
                avg_velocity = sender_history['Transaction_Velocity'].mean()
                if row[feature] > avg_velocity * 1.5:
                    reasons.append(f"High Transaction Velocity ({row[feature]:.2f}, {row[feature]/avg_velocity:.1f}x average)")
            elif feature == 'Unusual_Time' and row[feature]:
                reasons.append("Unusual Transaction Time")
            elif feature == 'Multiple_Currency_Conversions' and row[feature]:
                reasons.append("Multiple Currency Conversions")
            elif feature in ['Sender_Country', 'Receiver_Country'] and row['Sender_Country'] != row['Receiver_Country']:
                reasons.append(f"Cross-border Transaction ({row['Sender_Country']} to {row['Receiver_Country']})")
            elif feature == 'Is_Known_Fraudster' and row[feature]:
                reasons.append("Known Fraudster Involved")
            elif feature == 'Is_Sanctioned_Entity' and row[feature]:
                reasons.append("Sanctioned Entity Involved")
            elif 'Payment_Method_Cryptocurrency' in feature and row[feature]:
                reasons.append("Cryptocurrency Transaction")
            elif feature == 'IP_Address_Change' and row[feature]:
                reasons.append("Frequent IP Address Changes")
            elif feature == 'Device_Change' and row[feature]:
                reasons.append("Multiple Devices Used")
            elif feature == 'VPN_Usage' and row[feature]:
                reasons.append("VPN Usage Detected")
    
    if not reasons:
        if row['Suspicion_Score'] > 0.7:
            reasons.append(f"High overall suspicion score ({row['Suspicion_Score']:.2f})")
        elif row['Suspicion_Score'] > 0.5:
            reasons.append(f"Moderate overall suspicion score ({row['Suspicion_Score']:.2f})")
    
    return ', '.join(set(reasons)) if reasons else "No specific reason identified"

def classify_fraud_type(row, threshold: float = 0.5) -> str:
    """Assign a primary fraud type label based on rules and model score.

    Priority order ensures a single, consistent primary classification per transaction.
    """
    # High-level categories (ordered by priority)
    is_known_fraud = bool(row.get('Is_Known_Fraudster', False))
    is_sanctioned = bool(row.get('Is_Sanctioned_Entity', False))
    is_cross_border = row.get('Sender_Country', 'Unknown') != row.get('Receiver_Country', 'Unknown')
    is_crypto = any(k in row.index and row[k] for k in ['Payment_Method_Cryptocurrency', 'Payment_Method_Crypto', 'Payment_Method_crypto'])
    high_amount = float(row.get('Transaction_Amount', 0)) >= 10000
    high_velocity = float(row.get('Transaction_Velocity', 0)) >= 8
    unusual_time = bool(row.get('Unusual_Time', False))
    multi_fx = bool(row.get('Multiple_Currency_Conversions', False))
    vpn = bool(row.get('VPN_Usage', False))
    ip_change = bool(row.get('IP_Address_Change', False))
    new_device = bool(row.get('Is_New_Device', False) or row.get('Device_Change', False))
    new_location = bool(row.get('Is_New_Location', False))
    failed_attempts = int(row.get('Repeated_Failed_Attempts', 0)) >= 3
    score = float(row.get('Suspicion_Score', 0.0))

    # Primary classification based on priority
    if is_known_fraud:
        return 'Known Fraudster'
    if is_sanctioned:
        return 'Sanctioned Entity'
    if is_crypto and is_cross_border and (high_amount or multi_fx):
        return 'Crypto Cross-border Laundering'
    if is_cross_border and multi_fx:
        return 'Cross-border FX Risk'
    if high_amount and high_velocity:
        return 'Rapid Large Transfers'
    if vpn or ip_change:
        return 'Anonymity/Device Evasion'
    if new_device or new_location:
        return 'Account Takeover (ATO) Risk'
    if failed_attempts:
        return 'Credential Stuffing/Brute Force'
    if unusual_time:
        return 'Odd-hour Activity'
    if score >= max(0.8, threshold):
        return 'High-Risk (Model)'
    if score >= threshold:
        return 'Medium-Risk (Model)'
    return 'Low-Risk' 