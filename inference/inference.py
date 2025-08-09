import pandas as pd
import numpy as np
import joblib
import json

class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load('data/model.pkl')
        self.scaler = joblib.load('data/scaler.pkl')
        
        # Load feature names from training
        X_test = pd.read_csv('data/X_test.csv')
        self.feature_names = X_test.columns.tolist()
    
    def preprocess(self, data):

        df = pd.DataFrame([data])
        
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        
 
        binary_cols = ['gender', 'partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        
        cat_cols = ['multiplelines', 'internetservice', 'onlinesecurity',
                    'onlinebackup', 'deviceprotection', 'techsupport',
                    'streamingtv', 'streamingmovies', 'contract', 'paymentmethod']
        
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features used in training
        df = df[self.feature_names]
        
        return df
    
    def predict(self, customer_data):
        features = self.preprocess(customer_data)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': float(probability[1]),
            'retention_probability': float(probability[0])
        }

if __name__ == "__main__":
    print("🔮 Testing Churn Prediction...")
    
    predictor = ChurnPredictor()
    
    # Sample customer
    sample_customer = {
        'gender': 'Male',
        'seniorcitizen': 0,
        'partner': 'Yes',
        'dependents': 'No',
        'tenure': 12,
        'phoneservice': 'Yes',
        'multiplelines': 'No',
        'internetservice': 'DSL',
        'onlinesecurity': 'No',
        'onlinebackup': 'Yes',
        'deviceprotection': 'No',
        'techsupport': 'No',
        'streamingtv': 'No',
        'streamingmovies': 'No',
        'contract': 'Month-to-month',
        'paperlessbilling': 'Yes',
        'paymentmethod': 'Electronic check',
        'monthlycharges': 29.85,
        'totalcharges': 358.2
    }
    
    result = predictor.predict(sample_customer)
    
    print("\n📊 Prediction Results:")
    print(f"   Churn Prediction: {result['churn_prediction']}")
    print(f"   Churn Probability: {result['churn_probability']:.2%}")
    print(f"   Retention Probability: {result['retention_probability']:.2%}")
    
    # customer (likely to stay)
    loyal_customer = {
        'gender': 'Female',
        'seniorcitizen': 0,
        'partner': 'Yes',
        'dependents': 'Yes',
        'tenure': 72,
        'phoneservice': 'Yes',
        'multiplelines': 'Yes',
        'internetservice': 'Fiber optic',
        'onlinesecurity': 'Yes',
        'onlinebackup': 'Yes',
        'deviceprotection': 'Yes',
        'techsupport': 'Yes',
        'streamingtv': 'Yes',
        'streamingmovies': 'Yes',
        'contract': 'Two year',
        'paperlessbilling': 'No',
        'paymentmethod': 'Bank transfer (automatic)',
        'monthlycharges': 85.0,
        'totalcharges': 6000.0
    }
    
    result2 = predictor.predict(loyal_customer)
    
    print("\n Loyal Customer Prediction:")
    print(f"   Churn Prediction: {result2['churn_prediction']}")
    print(f"   Churn Probability: {result2['churn_probability']:.2%}")
    print(f"   Retention Probability: {result2['retention_probability']:.2%}")
    
    print("\n Inference test complete!")