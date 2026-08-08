import logging

from fastapi import FastAPI, HTTPException

from shared.schemas import EvaluateRequest, EvaluateResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("evaluation")

app = FastAPI(title="Evaluation Service")

from shared.metrics import instrument
instrument(app, service_name="evaluation")

DEFAULT_PRIMARY_METRIC = {
    "classification": "f1_weighted",
    "regression": "r2",
}
# Higher is better for these; lower is better for these (e.g. rmse, mae)
LOWER_IS_BETTER = {"rmse", "mae"}

QUALITY_BAR = {
    "classification": {"f1_weighted": 0.55},
    "regression": {"r2": 0.3},
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if not req.results:
        raise HTTPException(422, "No model results to evaluate")

    metric = req.primary_metric or DEFAULT_PRIMARY_METRIC[req.task_type]
    lower_better = metric in LOWER_IS_BETTER

    scored = [r for r in req.results if metric in r.metrics]
    if not scored:
        raise HTTPException(422, f"None of the results contain metric '{metric}'")

    best = min(scored, key=lambda r: r.metrics[metric]) if lower_better \
        else max(scored, key=lambda r: r.metrics[metric])

    leaderboard = sorted(
        scored, key=lambda r: r.metrics[metric], reverse=not lower_better
    )

    bar = QUALITY_BAR.get(req.task_type, {}).get(metric)
    if bar is None:
        passed = True
        reason = f"No quality bar configured for metric '{metric}'; accepting best result."
    elif lower_better:
        passed = best.metrics[metric] <= bar
        reason = (f"Best {metric}={best.metrics[metric]:.4f} "
                   f"{'meets' if passed else 'misses'} bar (<= {bar}).")
    else:
        passed = best.metrics[metric] >= bar
        reason = (f"Best {metric}={best.metrics[metric]:.4f} "
                   f"{'meets' if passed else 'misses'} bar (>= {bar}).")

    log.info("Evaluated run=%s best=%s passed=%s", req.run_id, best.model_name, passed)

    return EvaluateResponse(
        run_id=req.run_id,
        best_model_name=best.model_name,
        best_model_key=best.model_key,
        leaderboard=leaderboard,
        passed_quality_bar=passed,
        reason=reason,
    )
