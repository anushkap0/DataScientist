import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .graph import GRAPH
from shared.storage import get_bytes
from prometheus_client import Counter, Histogram

PIPELINE_RUNS = Counter(
    "pipeline_runs_total", "Completed pipeline runs by outcome", ["outcome"]
)
PIPELINE_DURATION = Histogram(
    "pipeline_run_duration_seconds", "End-to-end pipeline run duration",
    buckets=(5, 15, 30, 60, 120, 300, 600, 1800),
)
PIPELINE_RETRIES = Histogram(
    "pipeline_retries", "Number of feature/train retries before completion",
    buckets=(0, 1, 2, 3, 4, 5),
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orchestrator")

app = FastAPI(title="Autonomous Data Scientist — Orchestrator")

from shared.metrics import instrument
instrument(app, service_name="orchestrator")

# In-memory run registry. For a real multi-replica deployment, back this
# with Redis/Postgres so any orchestrator pod can serve /status for any
# run_id. Left simple here since orchestrator runs as a single replica
# by default (see k8s/orchestrator-deployment.yaml).
RUNS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=4)


class RunRequest(BaseModel):
    source_url: Optional[str] = None
    raw_csv_key: Optional[str] = None
    target_column: str
    max_retries: int = 2


class RunAck(BaseModel):
    run_id: str
    status: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _execute(run_id: str, initial_state: dict):
    RUNS[run_id]["status"] = "running"
    start = time.perf_counter()
    try:
        final_state = GRAPH.invoke(initial_state)
        RUNS[run_id]["status"] = "completed"
        RUNS[run_id]["result"] = final_state
        PIPELINE_RUNS.labels(outcome="success").inc()
        PIPELINE_RETRIES.observe(final_state.get("retries", 0))
    except Exception as e:
        log.exception("Run %s failed", run_id)
        RUNS[run_id]["status"] = "failed"
        RUNS[run_id]["error"] = str(e)
        PIPELINE_RUNS.labels(outcome="failure").inc()
    finally:
        PIPELINE_DURATION.observe(time.perf_counter() - start)


@app.post("/run", response_model=RunAck)
def run_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    if not req.source_url and not req.raw_csv_key:
        raise HTTPException(400, "Provide source_url or raw_csv_key")

    run_id = str(uuid.uuid4())
    initial_state = {
        "run_id": run_id,
        "source_url": req.source_url,
        "raw_csv_key": req.raw_csv_key,
        "target_column": req.target_column,
        "retries": 0,
        "max_retries": req.max_retries,
        "log": [],
    }
    RUNS[run_id] = {"status": "queued"}
    background_tasks.add_task(_execute, run_id, initial_state)
    return RunAck(run_id=run_id, status="queued")


@app.get("/status/{run_id}")
def status(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Unknown run_id")
    return RUNS[run_id]


@app.get("/artifacts/{run_id}/{name}")
def artifact(run_id: str, name: str):
    from fastapi.responses import Response
    try:
        data = get_bytes(f"runs/{run_id}/{name}")
    except Exception:
        raise HTTPException(404, "Artifact not found")
    return Response(content=data, media_type="application/octet-stream")
