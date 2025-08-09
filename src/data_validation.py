import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGED_DATA_PATH = "data/staged/data.csv"

def validate_data():
    df = pd.read_csv(STAGED_DATA_PATH)
    
    # Schema validation
    expected_columns = [
        'customerid', 'gender', 'seniorcitizen', 'partner', 'dependents',
        'tenure', 'phoneservice', 'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod', 'monthlycharges', 'totalcharges', 'churn'
    ]
    
    # Convert numpy types to Python native types for JSON serialization
    missing_values = df.isnull().sum()
    missing_dict = {str(k): int(v) for k, v in missing_values.items() if v > 0}
    
    # Basic validations
    validation_results = {
        'n_records': int(len(df)),
        'n_features': int(len(df.columns)),
        'missing_values': missing_dict,
        'duplicate_rows': int(df.duplicated().sum()),
        'schema_valid': set(df.columns) == set(expected_columns)
    }
    
    # Check TotalCharges (often has issues)
    if df['totalcharges'].dtype == 'object':
        logger.warning("TotalCharges is object type - needs conversion in feature engineering")
        validation_results['totalcharges_type'] = 'object - needs fixing'
    
    # Check for churn distribution
    if 'churn' in df.columns:
        churn_dist = df['churn'].value_counts()
        validation_results['churn_distribution'] = {
            str(k): int(v) for k, v in churn_dist.items()
        }
    
    # Log results
    logger.info(f"Validation Results: {validation_results}")
    
    # Save validation report
    with open('data/validation_report.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"✅ Data validation complete!")
    print(f"   Records: {validation_results['n_records']}")
    print(f"   Features: {validation_results['n_features']}")
    print(f"   Missing values: {len(validation_results['missing_values'])} columns")
    print(f"   Duplicate rows: {validation_results['duplicate_rows']}")
    
    return validation_results

if __name__ == "__main__":
    validate_data()