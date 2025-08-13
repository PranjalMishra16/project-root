from flask import Flask, Response
import json
from datetime import datetime

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    
    # Load your actual model metrics
    with open('data/metrics.json', 'r') as f:
        model_metrics = json.load(f)
    

    metrics_text = f"""

model_accuracy {model_metrics['accuracy']}

model_roc_auc {model_metrics['roc_auc']}

model_precision {model_metrics['precision']}

model_recall {model_metrics['recall']}

predictions_total {model_metrics.get('total_predictions', 1409)}
"""
    return Response(metrics_text, mimetype='text/plain')

@app.route('/')
def home():
    return "Model Metrics Server - Go to /metrics"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)