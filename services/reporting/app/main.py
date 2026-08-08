import logging
import os

from fastapi import FastAPI, HTTPException

from . import llm
from shared.schemas import ReportRequest, ReportResponse
from shared.storage import get_json, put_bytes, run_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reporting")

app = FastAPI(title="Reporting Service")

from shared.metrics import instrument
instrument(app, service_name="reporting")


@app.get("/health")
def health():
    return {"status": "ok", "provider": os.getenv("REPORT_LLM_PROVIDER", "anthropic")}


def _fallback_report(req: ReportRequest, eda: dict) -> str:
    """Deterministic report used if the LLM call fails, so the pipeline never dead-ends."""
    lines = [f"# Data Science Report — run `{req.run_id}`", ""]
    lines.append(f"**Task type:** {req.task_type}")
    lines.append(f"**Best model:** {req.best_model_name}")
    lines.append("")
    lines.append("## EDA highlights")
    for w in eda.get("warnings", []):
        lines.append(f"- {w}")
    lines.append("")
    lines.append("## Model leaderboard")
    for r in req.leaderboard:
        lines.append(f"- {r.model_name}: {r.metrics}")
    return "\n".join(lines)


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    try:
        eda = get_json(req.eda_report_key)
    except Exception as e:
        raise HTTPException(404, f"Could not load eda_report_key: {e}")

    prompt = f"""You are a data scientist writing a concise internal report.

Task type: {req.task_type}
EDA findings (JSON): {eda}
Model leaderboard (best first): {[r.model_dump() for r in req.leaderboard]}
Best model selected: {req.best_model_name}
Extra notes: {req.narrative_notes or "none"}

Write a markdown report with sections: Summary, Data Quality Notes, Modeling
Approach, Results, Recommendation. Be specific about numbers, be honest about
limitations, and keep it under 500 words."""

    try:
        report_md = llm.generate(prompt)
    except Exception as e:
        log.warning("LLM report generation failed (%s); using fallback report", e)
        report_md = _fallback_report(req, eda)

    report_key = run_key(req.run_id, "report.md")
    put_bytes(report_key, report_md.encode("utf-8"))

    log.info("Report generated run=%s provider=%s", req.run_id, os.getenv("REPORT_LLM_PROVIDER", "anthropic"))

    return ReportResponse(
        run_id=req.run_id,
        report_key=report_key,
        report_url_hint=f"GET /artifacts/{report_key} via orchestrator",
    )
