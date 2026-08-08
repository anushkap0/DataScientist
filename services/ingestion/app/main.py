import io
import logging

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException

from shared.schemas import IngestRequest, IngestResponse
from shared.storage import put_bytes, put_dataframe, run_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ingestion")

app = FastAPI(title="Ingestion Service")

from shared.metrics import instrument
instrument(app, service_name="ingestion")


@app.get("/health")
def health():
    return {"status": "ok"}


def _infer_task_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        return "regression"
    return "classification"


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if req.source_url:
        try:
            resp = httpx.get(req.source_url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
        except Exception as e:
            raise HTTPException(400, f"Could not fetch/parse source_url: {e}")
    elif req.raw_csv_key:
        from shared.storage import get_bytes
        try:
            df = pd.read_csv(io.BytesIO(get_bytes(req.raw_csv_key)))
        except Exception as e:
            raise HTTPException(400, f"Could not read raw_csv_key: {e}")
    else:
        raise HTTPException(400, "Provide either source_url or raw_csv_key")

    if req.target_column not in df.columns:
        raise HTTPException(
            422, f"target_column '{req.target_column}' not found in columns {list(df.columns)}"
        )

    if df.empty:
        raise HTTPException(422, "Dataset is empty")

    raw_key = run_key(req.run_id, "raw.csv")
    put_dataframe(raw_key, df)

    task_type = _infer_task_type(df[req.target_column])

    log.info("Ingested run=%s rows=%d cols=%d task_type=%s", req.run_id, len(df), len(df.columns), task_type)

    return IngestResponse(
        run_id=req.run_id,
        raw_data_key=raw_key,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        task_type=task_type,
    )
