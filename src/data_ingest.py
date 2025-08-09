import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (embedded since no configs folder)
RAW_DATA_PATH = "data/raw/telco_customer_churn.csv"
STAGED_DATA_PATH = "data/staged/data.csv"

def ingest_data():
    # Read raw data
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df)} records from {RAW_DATA_PATH}")
    
    # Basic cleaning - lowercase column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Save to staged
    Path("data/staged").mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGED_DATA_PATH, index=False)
    
    logger.info(f"Saved staged data to {STAGED_DATA_PATH}")
    print(f"✅ Data ingestion complete! Shape: {df.shape}")

if __name__ == "__main__":
    ingest_data()