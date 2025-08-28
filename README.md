# 🚀 Telco Customer Churn MLOps Pipeline

## 📋 Project Overview

This project implements a production-grade MLOps pipeline for predicting customer churn in telecommunications. It demonstrates end-to-end machine learning operations from data versioning to model deployment and monitoring.

### 🎯 Problem Statement
- **Goal**: Predict which customers are likely to churn (cancel service)
- **Dataset**: Telco Customer Churn dataset (7,043 customers, 21 features)
- **Business Impact**: Early identification of at-risk customers for retention campaigns

### 📊 Model Performance
- **ROC-AUC**: 0.842
- **Accuracy**: 80.06%
- **Precision**: 65.98%
- **Recall**: 51.34%
- **F1 Score**: 57.74%

Run ML Pipeline
bash# Run complete pipeline
dvc repro

# Or run individual stages
python src/data_ingest.py
python src/data_validation.py
python src/train_and_tune.py
python src/evaluate.py

Check Metrics
bashdvc metrics show
cat data/metrics.json

Test Inference
bashpython inference/inference.py

🐳 Docker
Build Image
bashdocker build -t project-root .

Run Container
bashdocker run project-root

Start Monitoring Stack
bash# Start Prometheus & Grafana
docker-compose up -d

# Access services
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
