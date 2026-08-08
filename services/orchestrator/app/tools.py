"""
Thin, typed HTTP clients the LangGraph nodes use to call each microservice.
Centralizing them here means the graph nodes stay readable (no raw httpx
calls scattered around) and service URLs are configured in exactly one
place, via env vars — each pointing at the corresponding K8s Service DNS
name (e.g. http://ingestion-svc:8000).
"""
import os

import httpx

from shared.schemas import (
    EDARequest,
    EDAResponse,
    EvaluateRequest,
    EvaluateResponse,
    FeatureEngineeringRequest,
    FeatureEngineeringResponse,
    IngestRequest,
    IngestResponse,
    ReportRequest,
    ReportResponse,
    TrainRequest,
    TrainResponse,
)

INGESTION_URL = os.getenv("INGESTION_URL", "http://ingestion-svc:8000")
EDA_URL = os.getenv("EDA_URL", "http://eda-svc:8000")
FEATURE_ENGINEERING_URL = os.getenv("FEATURE_ENGINEERING_URL", "http://feature-engineering-svc:8000")
TRAINING_URL = os.getenv("TRAINING_URL", "http://training-svc:8000")
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://evaluation-svc:8000")
REPORTING_URL = os.getenv("REPORTING_URL", "http://reporting-svc:8000")

TIMEOUT = httpx.Timeout(120.0)


def _post(url: str, payload: dict) -> dict:
    resp = httpx.post(url, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def call_ingest(req: IngestRequest) -> IngestResponse:
    return IngestResponse(**_post(f"{INGESTION_URL}/ingest", req.model_dump()))


def call_eda(req: EDARequest) -> EDAResponse:
    return EDAResponse(**_post(f"{EDA_URL}/analyze", req.model_dump()))


def call_feature_engineering(req: FeatureEngineeringRequest) -> FeatureEngineeringResponse:
    return FeatureEngineeringResponse(**_post(f"{FEATURE_ENGINEERING_URL}/transform", req.model_dump()))


def call_train(req: TrainRequest) -> TrainResponse:
    return TrainResponse(**_post(f"{TRAINING_URL}/train", req.model_dump()))


def call_evaluate(req: EvaluateRequest) -> EvaluateResponse:
    return EvaluateResponse(**_post(f"{EVALUATION_URL}/evaluate", req.model_dump()))


def call_report(req: ReportRequest) -> ReportResponse:
    return ReportResponse(**_post(f"{REPORTING_URL}/report", req.model_dump()))
