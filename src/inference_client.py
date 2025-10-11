import argparse
import boto3

# >>> Replace with your deployed endpoint name <<<
ENDPOINT_NAME = "REPLACE_WITH_YOUR_ENDPOINT_NAME"

def main(payload: str):
    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",  # default for sagemaker-scikit-learn
        Body=payload.encode("utf-8")
    )
    body = response["Body"].read().decode("utf-8")
    print("Prediction:", body)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help='CSV line, e.g. "1200,3"')
    args = parser.parse_args()
    main(args.payload)
