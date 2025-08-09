import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_DATA_PATH = "data/staged/data.csv"
MODEL_PATH = "data/model.pkl"
SCALER_PATH = "data/scaler.pkl"

def prepare_features(df):
    """Prepare features for training"""
    # Handle TotalCharges - fixed the warning
    if 'totalcharges' in df.columns:
        df['totalcharges'] = df['totalcharges'].replace(' ', np.nan)
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        # Fixed: use loc instead of chained assignment
        df.loc[:, 'totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())
    
    
    # Drop customerID
    if 'customerid' in df.columns:
        df = df.drop('customerid', axis=1)
    
    # Encode binary columns
    binary_cols = ['gender', 'partner', 'dependents', 'phoneservice', 
                   'paperlessbilling', 'churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    
    # One-hot encode categorical columns
    cat_cols = ['multiplelines', 'internetservice', 'onlinesecurity',
                'onlinebackup', 'deviceprotection', 'techsupport',
                'streamingtv', 'streamingmovies', 'contract', 'paymentmethod']
    
    # Only encode columns that exist
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def train_and_tune():
    # Load data
    df = pd.read_csv(STAGED_DATA_PATH)
    
    # Prepare features
    df = prepare_features(df)
    
    # Split features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning (smaller grid for faster training)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    logger.info("Starting hyperparameter tuning...")
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Save model and scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save test data for evaluation
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    # Save best parameters (convert numpy types to Python types)
    best_params_clean = {
        k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
        for k, v in grid_search.best_params_.items()
    }
    
    with open('data/best_params.json', 'w') as f:
        json.dump(best_params_clean, f, indent=2)
    
    print(f"✅ Training complete!")
    print(f"   Best CV Score: {grid_search.best_score_:.4f}")
    print(f"   Best Params: {best_params_clean}")
    
    return best_model

if __name__ == "__main__":
    train_and_tune()