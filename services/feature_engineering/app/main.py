import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import embeddings
from shared.schemas import FeatureEngineeringRequest, FeatureEngineeringResponse
from shared.storage import get_dataframe, put_dataframe, run_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("feature_engineering")

app = FastAPI(title="Feature Engineering Service")

from shared.metrics import instrument
instrument(app, service_name="feature_engineering")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transform", response_model=FeatureEngineeringResponse)
def transform(req: FeatureEngineeringRequest):
    try:
        df = get_dataframe(req.raw_data_key)
    except Exception as e:
        raise HTTPException(404, f"Could not load raw_data_key: {e}")

    if req.target_column not in df.columns:
        raise HTTPException(422, f"target_column '{req.target_column}' not in data")

    df = df.drop(columns=[c for c in req.drop_columns if c in df.columns])

    # Drop obvious ID-like high-cardinality string columns (not the target)
    for col in df.select_dtypes(include=["object"]).columns:
        if col != req.target_column and df[col].nunique() > 0.9 * len(df):
            df = df.drop(columns=[col])

    df = df.dropna(subset=[req.target_column])
    y = df[req.target_column]
    X = df.drop(columns=[req.target_column])

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Free-text columns (reviews, descriptions, comments) get Hugging Face
    # sentence embeddings instead of label encoding, since squashing "great
    # product, fast delivery" into a single integer throws away the signal.
    # Short categorical strings ("month-to-month", "yes"/"no") still just
    # get label-encoded below — embeddings would be overkill and slower.
    embedded_cols = []
    label_encode_cols = []
    for col in categorical_cols:
        if embeddings.is_free_text_column(X[col]):
            embedded_cols.append(col)
        else:
            label_encode_cols.append(col)

    if numeric_cols:
        X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])
        X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

    for col in label_encode_cols:
        X[col] = X[col].fillna("__missing__")
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    embedding_frames = []
    for col in embedded_cols:
        try:
            emb_df = embeddings.embed_column(X[col], prefix=col)
            embedding_frames.append(emb_df)
            X = X.drop(columns=[col])
            log.info("Embedded free-text column '%s' -> %d dims via Hugging Face", col, emb_df.shape[1])
        except Exception as e:
            # Never let an unavailable embedding model fail the whole
            # pipeline — fall back to label encoding for this column.
            log.warning("HF embedding failed for column '%s' (%s); falling back to label encoding", col, e)
            X[col] = X[col].fillna("__missing__")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if embedding_frames:
        X = pd.concat([X] + embedding_frames, axis=1)

    if req.task_type == "classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    out = X.copy()
    out[req.target_column] = y

    features_key = run_key(req.run_id, "features.csv")
    put_dataframe(features_key, out)

    log.info("Feature engineering complete run=%s n_features=%d", req.run_id, X.shape[1])

    return FeatureEngineeringResponse(
        run_id=req.run_id,
        features_key=features_key,
        feature_names=list(X.columns),
        n_rows=len(out),
    )
