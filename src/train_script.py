# train_script.py
# -----------------------------------------------------------------------------
# SageMaker SKLearn training + inference script:
# - trains on CSV in /opt/ml/input/data/train
# - saves model to /opt/ml/model
# - implements model_fn, input_fn, predict_fn, output_fn
# -----------------------------------------------------------------------------

import argparse, json, os, sys, joblib, logging
from typing import List
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def _read_csv_auto(path: str, sep: str = ",", header: str = "infer") -> pd.DataFrame:
    header_arg = "infer" if header == "infer" else None
    return pd.read_csv(path, sep=sep, header=header_arg)

def _split_features_target(df: pd.DataFrame, target_col: str = None):
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        feat = list(X.columns)
        logger.info(f"Use target column: {target_col}")
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        feat = list(X.columns)
        if target_col:
            logger.warning(f"'{target_col}' not found; fallback to last column '{df.columns[-1]}'")
        else:
            logger.info(f"No target-col specified; use last column '{df.columns[-1]}'")
    return X, y, feat

def _build_model(algorithm: str, n_estimators: int, random_state: int):
    algorithm = algorithm.lower()
    if algorithm in ("linear", "linreg", "linear_regression"):
        return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    if algorithm in ("rf", "random_forest", "randomforest"):
        return Pipeline([("scaler", StandardScaler()),
                         ("regressor", RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1))])
    raise ValueError("Unknown algorithm. Use 'linear' or 'random_forest'.")

def train(args):
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    out_dir   = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    os.makedirs(model_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    if args.train_file:
        train_path = os.path.join(train_dir, args.train_file)
    else:
        csvs = [f for f in os.listdir(train_dir) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV under {train_dir}")
        train_path = os.path.join(train_dir, csvs[0])

    logger.info(f"Reading: {train_path}")
    df = _read_csv_auto(train_path, sep=args.sep, header=args.header)
    logger.info(f"Shape: {df.shape}, columns: {list(df.columns)}")

    X, y, feat = _split_features_target(df, args.target_col)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    model = _build_model(args.algorithm, args.n_estimators, args.random_state)
    model.fit(Xtr, ytr)

    ypred = model.predict(Xva)
    metrics = {"rmse": float(mean_squared_error(yva, ypred, squared=False)),
               "r2": float(r2_score(yva, ypred))}
    logger.info(f"Validation metrics: {metrics}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump(metrics, f)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "columns.json"), "w") as f:
        json.dump({"feature_names": feat, "target_col": args.target_col or "__last_col__"}, f)
    logger.info("Artifacts saved.")

# ---------------- Inference hooks ----------------
def model_fn(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    schema = os.path.join(model_dir, "columns.json")
    if os.path.exists(schema):
        with open(schema) as f: model._feature_names = json.load(f).get("feature_names")
    else:
        model._feature_names = None
    return model

def input_fn(request_body: str, content_type: str):
    if content_type == "text/csv":
        data = pd.read_csv(pd.compat.StringIO(request_body), header=None)
        return data
    if content_type.startswith("application/json"):
        body = json.loads(request_body)
        if "instances" in body:
            arr = np.array(body["instances"])
        elif "data" in body:
            arr = np.array([body["data"]])
        else:
            arr = np.array(body)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        return pd.DataFrame(arr)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: pd.DataFrame, model):
    feat = getattr(model, "_feature_names", None)
    if feat is not None and list(input_data.columns) != feat:
        input_data = input_data.copy(); input_data.columns = feat
    return model.predict(input_data)

def output_fn(prediction: np.ndarray, accept: str):
    if accept.startswith("application/json"):
        return json.dumps({"predictions": prediction.tolist()}), "application/json"
    if accept == "text/csv":
        return "\n".join(map(str, prediction.tolist())), "text/csv"
    return json.dumps({"predictions": prediction.tolist()}), "application/json"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=str, default="")
    p.add_argument("--sep", type=str, default=",")
    p.add_argument("--header", type=str, default="infer", choices=["infer", "none"])
    p.add_argument("--target-col", type=str, default="price")
    p.add_argument("--algorithm", type=str, default="linear")  # linear | random_forest
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.info(f"Args: {vars(args)}")
    train(args)
