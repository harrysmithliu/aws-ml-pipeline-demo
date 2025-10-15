import argparse
import boto3
from botocore.exceptions import ClientError

def safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        # 忽略常见的不存在错误
        if code not in ("ValidationException", "ResourceNotFound", "ResourceNotFoundException"):
            raise

def main(endpoint: str, delete_model: bool = False):
    sm = boto3.client("sagemaker")

    # 1) 拿到 EndpointConfigName
    cfg_name = None
    try:
        info = sm.describe_endpoint(EndpointName=endpoint)
        cfg_name = info.get("EndpointConfigName")
    except ClientError:
        pass

    # 2) 删 Endpoint
    print(f"Deleting endpoint: {endpoint}")
    safe(sm.delete_endpoint, EndpointName=endpoint)

    # 3) 删 Endpoint Config（如果存在）
    if cfg_name:
        print(f"Deleting endpoint config: {cfg_name}")
        safe(sm.delete_endpoint_config, EndpointConfigName=cfg_name)

        # 4) 可选：删 Model（从 Endpoint Config 的 production variants 中找）
        if delete_model:
            try:
                ec = sm.describe_endpoint_config(EndpointConfigName=cfg_name)
                models = {pv["ModelName"] for pv in ec.get("ProductionVariants", [])}
                for m in models:
                    print(f"Deleting model: {m}")
                    safe(sm.delete_model, ModelName=m)
            except ClientError:
                pass

    print("Teardown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Endpoint name")
    parser.add_argument("--delete-model", action="store_true", help="Also delete model(s) bound to the endpoint config")
    args = parser.parse_args()
    main(args.endpoint, args.delete_model)
