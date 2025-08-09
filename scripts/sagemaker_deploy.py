import json
from datetime import datetime

# SageMaker deployment configuration
deployment_config = {
    "timestamp": str(datetime.now()),
    "model_name": "project-root-model",
    "endpoint_name": "project-root-endpoint",
    "ecr_image": "752953536471.dkr.ecr.us-east-1.amazonaws.com/project-root:latest",
    "instance_type": "ml.t2.medium",
    "initial_instance_count": 1,
    "deployment_status": "Ready for deployment"
}

print("="*60)
print("SAGEMAKER DEPLOYMENT CONFIGURATION")
print("="*60)
print(json.dumps(deployment_config, indent=2))
print("\nDeployment command:")
print("aws sagemaker create-model --model-name project-root-model \\")
print("  --primary-container Image=752953536471.dkr.ecr.us-east-1.amazonaws.com/project-root:latest \\")
print("  --execution-role-arn arn:aws:iam::752953536471:role/SageMakerRole")
print("\nEndpoint ready for creation via AWS Console or CLI")