FROM python:3.9-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install pandas numpy scikit-learn joblib scipy

# Verify files exist in container
RUN ls -la data/

CMD ["python", "inference/inference.py"]