import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "data/model.pkl"
SCALER_PATH = "data/scaler.pkl"

def evaluate():
    """Evaluate the trained model"""
    
    # Check if required files exist
    if not Path(MODEL_PATH).exists():
        print("❌ Model not found! Run train_and_tune.py first")
        return None
    
    if not Path(SCALER_PATH).exists():
        print("❌ Scaler not found! Run train_and_tune.py first")
        return None
    
    if not Path('data/X_test.csv').exists() or not Path('data/y_test.csv').exists():
        print("❌ Test data not found! Run train_and_tune.py first")
        return None
    
    # Load model and test data
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    # Calculate additional metrics
    total_predictions = len(y_test)
    metrics['total_predictions'] = total_predictions
    metrics['predicted_churn'] = int(sum(y_pred))
    metrics['actual_churn'] = int(sum(y_test))
    metrics['churn_rate'] = float(sum(y_test) / len(y_test))
    
    # Save metrics
    with open('data/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print detailed results
    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE!")
    print("="*50)
    
    print("\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\n🎯 CONFUSION MATRIX:")
    print(f"   True Negatives:  {metrics['true_negatives']}")
    print(f"   False Positives: {metrics['false_positives']}")
    print(f"   False Negatives: {metrics['false_negatives']}")
    print(f"   True Positives:  {metrics['true_positives']}")
    
    print("\n📈 PREDICTION SUMMARY:")
    print(f"   Total Customers:    {metrics['total_predictions']}")
    print(f"   Predicted to Churn: {metrics['predicted_churn']}")
    print(f"   Actually Churned:   {metrics['actual_churn']}")
    print(f"   Actual Churn Rate:  {metrics['churn_rate']:.2%}")
    
    # Generate classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Churn', 'Churn']))
    
    return metrics

if __name__ == "__main__":
    evaluate()