"""Pydantic contracts shared by every service. Keep these in lockstep —
this file is copied into each service's Docker image at build time (see
each Dockerfile), so it is the single source of truth for the wire format."""
from typing import Any, Literal, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    run_id: str
    source_url: Optional[str] = None          # http(s) URL to a CSV
    raw_csv_key: Optional[str] = None          # already-uploaded object key
    target_column: str


class IngestResponse(BaseModel):
    run_id: str
    raw_data_key: str
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    task_type: Literal["classification", "regression"]


class EDARequest(BaseModel):
    run_id: str
    raw_data_key: str
    target_column: str


class EDAResponse(BaseModel):
    run_id: str
    eda_report_key: str
    missing_pct: dict[str, float]
    numeric_summary: dict[str, dict[str, float]]
    high_cardinality_cols: list[str]
    warnings: list[str]


class FeatureEngineeringRequest(BaseModel):
    run_id: str
    raw_data_key: str
    target_column: str
    task_type: Literal["classification", "regression"]
    drop_columns: list[str] = []


class FeatureEngineeringResponse(BaseModel):
    run_id: str
    features_key: str
    feature_names: list[str]
    n_rows: int


class TrainRequest(BaseModel):
    run_id: str
    features_key: str
    target_column: str
    task_type: Literal["classification", "regression"]
    candidate_models: list[str] = []  # empty = service default set


class ModelResult(BaseModel):
    model_name: str
    model_key: str
    metrics: dict[str, float]


class TrainResponse(BaseModel):
    run_id: str
    results: list[ModelResult]


class EvaluateRequest(BaseModel):
    run_id: str
    results: list[ModelResult]
    task_type: Literal["classification", "regression"]
    primary_metric: Optional[str] = None


class EvaluateResponse(BaseModel):
    run_id: str
    best_model_name: str
    best_model_key: str
    leaderboard: list[ModelResult]
    passed_quality_bar: bool
    reason: str


class ReportRequest(BaseModel):
    run_id: str
    eda_report_key: str
    leaderboard: list[ModelResult]
    best_model_name: str
    task_type: Literal["classification", "regression"]
    narrative_notes: Optional[str] = None


class ReportResponse(BaseModel):
    run_id: str
    report_key: str
    report_url_hint: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None
