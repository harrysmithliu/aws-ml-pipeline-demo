# AWS ML Pipeline Demo (Free Tier)

A minimal end-to-end machine learning workflow using **Amazon SageMaker (SKLearn)**, built to train, deploy, and test a regression model fully within the **AWS Free Tier**.

End-to-end ML pipeline on **AWS Free Tier** using **S3 → SageMaker (training & endpoint) → CloudWatch**, with a local **FastAPI** client for testing.

> **Goal:** Be portfolio-ready in 3–6 days, stay within free tier, and keep costs at \$0 by shutting down resources when done.

---

## Architecture
```
S3 (dataset) ──► SageMaker (train) ──► S3 (model artifacts)
                                 └──► SageMaker Endpoint (real-time inference)
Local FastAPI client ──► invoke endpoint ──► CloudWatch (logs/metrics)
```
**Free tier**: S3 (≤5GB), SageMaker t2.micro notebook (≤250 hrs / month in first 12 months), basic CloudWatch.  
**Important:** After testing, **delete the endpoint** to avoid charges.

---

## What you get
- Minimal regression model (sklearn `LinearRegression`) trained in SageMaker
- Model artifacts auto-saved to S3
- One-click deployment to a **real-time endpoint**
- Python client (`inference_client.py`) to invoke the endpoint
- `teardown.py` to clean up endpoint/model
- Sample dataset under `data/house_prices.csv`
- **All within Free Tier** (if you follow the notes below)

---

## Prerequisites
- AWS account with Free Tier
- An **IAM role** with access to S3 & SageMaker (you can use the default SageMaker execution role)
- Python 3.10+ locally if you want to run the client
- Install CLI SDKs locally (optional):
  ```bash
  pip install -r src/requirements.txt
  ```

---

## Quick Start (recommended 3–6 day flow)

### Day 1 – Set up & upload data
1. Create an S3 bucket, e.g. `s3://<your-bucket-name>` in your region.
2. Upload `data/house_prices.csv` to `s3://<your-bucket-name>/datasets/house_prices.csv`.

### Day 2 – Launch SageMaker Notebook
1. In AWS Console → SageMaker → Notebook instances → **Create** (choose **t2.micro**).  
2. Attach an execution role with access to S3 & SageMaker.  
3. Open Jupyter and upload the files from this repo (`notebooks/train_model.ipynb`, `src/train_script.py`).

### Day 3 – Train in SageMaker
1. Open `notebooks/train_model.ipynb` and set:
   - `S3_BUCKET = "<your-bucket-name>"`
   - `S3_DATA_KEY = "datasets/house_prices.csv"`
2. Run the notebook cells to:
   - Upload data (if needed)
   - Launch a training job using the **SKLearn Estimator**
   - Produce **model artifacts** in S3

### Day 4 – Deploy real-time endpoint
- Continue the notebook to call `estimator.deploy(...)` and create a **real-time endpoint**.
- The default SKLearn container expects **CSV**. Example payload: `1200,3`

### Day 5 – Test with the local client
1. Set your endpoint name in `src/inference_client.py` (variable `ENDPOINT_NAME`).
2. Run locally:
   ```bash
   python src/inference_client.py --payload "1200,3"
   ```
3. Check **CloudWatch Logs** (SageMaker → Inference → Endpoints → Logs).

### Day 6 – Clean up (to keep it \$0)
- Run `python src/teardown.py --endpoint <your-endpoint> --model <your-model-name>`  
- Or delete endpoint/model from the SageMaker console.
- Stop/Terminate the **Notebook instance** when not in use.

---

## Project Layout
```
aws-ml-pipeline-demo/
├── data/
│   └── house_prices.csv               # sample small dataset
├── notebooks/
│   └── train_model.ipynb              # run inside SageMaker Notebook
├── src/
│   ├── train_script.py                # executed by SKLearn Estimator (SageMaker)
│   ├── inference_client.py            # local script to invoke the endpoint
│   ├── teardown.py                    # safely delete endpoint/model
│   └── requirements.txt               # local deps if you want to run client
└── README.md
```

---

## Payload Format (CSV)
The default SKLearn container uses CSV, so for a feature vector `[area, bedrooms]`:
```
1200,3
```

If you prefer JSON, you'd need a custom inference script with `input_fn`/`predict_fn`/`output_fn`—to keep this starter **simple and free**, we use CSV.

---

## Free Tier Safety Checklist
- **Always stop or delete**: Notebook instance, Endpoints, Training jobs when done.
- Keep S3 under **5GB**.
- Use **t2.micro** where possible.
- Watch **Billing → Free Tier usage** in Console.

---

## Notes
- You can replace the model with XGBoost / RandomForest by editing `src/train_script.py`.
- To switch to JSON payloads, add a custom inference module (advanced).
- For a nicer demo, add a small Streamlit or FastAPI UI on your laptop.

---

## Why this project matters
This demo demonstrates your ability to:
- Configure AWS SageMaker from scratch
- Train and deploy a machine learning model in the cloud
- Control cost and follow best practices (Free Tier safe)
- Build reproducible MLOps-style pipelines

Ideal for students and early-career developers exploring cloud ML deployment.

---

## Tech Highlights
- AWS SageMaker (Scikit-Learn)
- S3 dataset integration
- Batch Transform testing (no endpoint billing)
- CloudWatch monitoring
- Modular design for XGBoost / RandomForest / LinearRegression

### Prediction Output
| Living Area (sqft) | Bedrooms | Predicted Price (USD) |
| -----------------: | -------: | --------------------: |
|               1200 |        3 |               322,957 |
|               1500 |        4 |               390,672 |


Model trained and deployed using AWS SageMaker Scikit-Learn.  
Predictions generated offline via Jupyter Notebook and stored in S3.

**Author**: Harry Liu
**Use at your own risk. Remember to tear down cloud resources.**
