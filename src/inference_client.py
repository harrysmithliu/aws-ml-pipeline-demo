import boto3, json

endpoint = "your-endpoint-name"
runtime = boto3.client("sagemaker-runtime")

row = [1200, 3]  # example: living_area, bedrooms
payload = ",".join(map(str, row))

resp = runtime.invoke_endpoint(
    EndpointName=endpoint,
    ContentType="text/csv",
    Body=payload.encode("utf-8"),
    Accept="application/json"
)
print(json.loads(resp["Body"].read().decode("utf-8")))
