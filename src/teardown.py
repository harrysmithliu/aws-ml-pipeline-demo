import argparse
import boto3

def main(endpoint: str, model: str = None):
    sm = boto3.client("sagemaker")
    print(f"Deleting endpoint: {endpoint}")
    sm.delete_endpoint(EndpointName=endpoint)
    print(f"Deleting endpoint config for: {endpoint}")
    sm.delete_endpoint_config(EndpointConfigName=endpoint)

    if model:
        print(f"Deleting model: {model}")
        sm.delete_model(ModelName=model)
    print("Teardown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Endpoint name")
    parser.add_argument("--model", required=False, help="Model name (optional)")
    args = parser.parse_args()
    main(args.endpoint, args.model)
