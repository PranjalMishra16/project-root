FROM python:3.9-slim

WORKDIR /app

# Copy everything to the container
COPY . .

# Install Python packages
RUN pip install pandas numpy scikit-learn joblib

# Run the inference test
CMD ["python", "inference/inference.py"]