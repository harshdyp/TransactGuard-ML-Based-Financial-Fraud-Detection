from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import time
from datetime import datetime

app = FastAPI(title="FraudX API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    model = joblib.load('fraud_detection_model.joblib')
    scaler = joblib.load('fraud_detection_scaler.joblib')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    sender_id: str
    receiver_id: str
    sender_country: str
    receiver_country: str
    payment_method: str
    timestamp: Optional[str] = None

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_suspicious: bool
    confidence: float
    risk_factors: List[str]
    processing_time: float

def preprocess_transaction(transaction: TransactionRequest):
    """Preprocess transaction data for prediction"""
    # Create feature vector
    features = []
    
    # Basic features
    features.append(transaction.amount)
    features.append(len(transaction.sender_id))
    features.append(len(transaction.receiver_id))
    
    # Country features
    features.append(1 if transaction.sender_country == transaction.receiver_country else 0)
    
    # Payment method encoding
    payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'crypto', 'paypal']
    payment_encoding = [1 if transaction.payment_method == method else 0 for method in payment_methods]
    features.extend(payment_encoding)
    
    # Amount-based features
    features.append(np.log1p(transaction.amount))
    features.append(transaction.amount ** 2)
    
    # Risk indicators
    features.append(1 if transaction.amount > 10000 else 0)  # High amount
    features.append(1 if transaction.amount < 10 else 0)     # Low amount
    
    return np.array(features).reshape(1, -1)

def identify_risk_factors(transaction: TransactionRequest, fraud_score: float) -> List[str]:
    """Identify risk factors for the transaction"""
    risk_factors = []
    
    if fraud_score > 0.7:
        risk_factors.append("High fraud probability")
    
    if transaction.amount > 10000:
        risk_factors.append("High transaction amount")
    
    if transaction.amount < 10:
        risk_factors.append("Suspiciously low amount")
    
    if transaction.sender_country != transaction.receiver_country:
        risk_factors.append("Cross-border transaction")
    
    if transaction.payment_method in ['crypto', 'bank_transfer']:
        risk_factors.append("High-risk payment method")
    
    return risk_factors

def calculate_confidence(fraud_score: float) -> float:
    """Calculate confidence based on fraud score"""
    # Higher confidence for extreme scores
    if fraud_score > 0.8 or fraud_score < 0.2:
        return 0.95
    elif fraud_score > 0.6 or fraud_score < 0.4:
        return 0.85
    else:
        return 0.75

@app.post("/api/v1/detect-fraud", response_model=FraudResponse)
async def detect_fraud(transaction: TransactionRequest):
    start_time = time.time()
    
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Preprocess transaction
        features = preprocess_transaction(transaction)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        fraud_score = model.predict_proba(features_scaled)[0][1]
        
        # Determine risk factors
        risk_factors = identify_risk_factors(transaction, fraud_score)
        
        processing_time = time.time() - start_time
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            fraud_score=float(fraud_score),
            is_suspicious=fraud_score > 0.5,
            confidence=calculate_confidence(fraud_score),
            risk_factors=risk_factors,
            processing_time=processing_time
        )
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model is not None and scaler is not None
    }

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "total_requests": 0,  # TODO: Implement counter
        "avg_response_time": 0.0,  # TODO: Implement tracking
        "models_loaded": model is not None and scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/batch-detect")
async def batch_detect_fraud(transactions: List[TransactionRequest]):
    """Process multiple transactions at once"""
    results = []
    start_time = time.time()
    
    for transaction in transactions:
        try:
            result = await detect_fraud(transaction)
            results.append(result)
        except Exception as e:
            results.append({
                "transaction_id": transaction.transaction_id,
                "error": str(e),
                "fraud_score": 0.5,
                "is_suspicious": False,
                "confidence": 0.0,
                "risk_factors": [],
                "processing_time": 0.0
            })
    
    total_time = time.time() - start_time
    
    return {
        "results": results,
        "total_transactions": len(transactions),
        "total_processing_time": total_time,
        "avg_processing_time": total_time / len(transactions) if transactions else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




















