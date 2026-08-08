import logging

from fastapi import FastAPI, HTTPException

from shared.schemas import EDARequest, EDAResponse
from shared.storage import get_dataframe, put_json, run_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eda")

app = FastAPI(title="EDA Service")

from shared.metrics import instrument
instrument(app, service_name="eda")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=EDAResponse)
def analyze(req: EDARequest):
    try:
        df = get_dataframe(req.raw_data_key)
    except Exception as e:
        raise HTTPException(404, f"Could not load raw_data_key: {e}")

    warnings: list[str] = []

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    high_missing = [c for c, p in missing_pct.items() if p > 40]
    if high_missing:
        warnings.append(f"Columns with >40% missing values: {high_missing}")

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = {}
    for col in numeric_df.columns:
        desc = numeric_df[col].describe()
        numeric_summary[col] = {k: float(v) for k, v in desc.items()}

    high_cardinality_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() > 0.5 * len(df)
    ]
    if high_cardinality_cols:
        warnings.append(f"High-cardinality categorical columns (likely IDs): {high_cardinality_cols}")

    if req.target_column in df.columns:
        target_missing = df[req.target_column].isna().sum()
        if target_missing:
            warnings.append(f"Target column has {target_missing} missing values")
        if df[req.target_column].dtype == "object" and df[req.target_column].nunique() > 50:
            warnings.append("Target has very high cardinality for classification — verify task_type")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows found")

    report = {
        "shape": list(df.shape),
        "missing_pct": missing_pct,
        "numeric_summary": numeric_summary,
        "high_cardinality_cols": high_cardinality_cols,
        "duplicate_rows": duplicate_rows,
        "correlations_with_target": (
            numeric_df.corrwith(df[req.target_column]).round(3).dropna().to_dict()
            if req.target_column in numeric_df.columns
            else {}
        ),
        "warnings": warnings,
    }

    eda_key = run_key(req.run_id, "eda_report.json")
    put_json(eda_key, report)

    log.info("EDA complete run=%s warnings=%d", req.run_id, len(warnings))

    return EDAResponse(
        run_id=req.run_id,
        eda_report_key=eda_key,
        missing_pct=missing_pct,
        numeric_summary=numeric_summary,
        high_cardinality_cols=high_cardinality_cols,
        warnings=warnings,
    )
