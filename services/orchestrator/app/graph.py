"""
The agent graph. Each node calls exactly one microservice (or reasons
with the LLM) and only updates state — the graph itself is what makes
this "autonomous": it decides whether to retry feature engineering, retry
training with a different model set, or proceed to reporting, without a
human in the loop.

Flow:
    ingest -> eda -> feature_engineering -> train -> evaluate --+--> report -> END
                          ^                                     |
                          +----------- retry (if quality bar not met, retries left) -+
"""
import logging
import os

from langgraph.graph import END, StateGraph

from . import tools
from .state import PipelineState
from shared.schemas import (
    EDARequest,
    EvaluateRequest,
    FeatureEngineeringRequest,
    IngestRequest,
    ReportRequest,
    TrainRequest,
)

log = logging.getLogger("orchestrator.graph")

# Escalating candidate model sets used across retries, so a failed retry
# actually tries something different rather than repeating itself.
RETRY_MODEL_SETS = {
    "classification": [
        ["logistic_regression"],
        ["random_forest", "gradient_boosting"],
        ["logistic_regression", "random_forest", "gradient_boosting"],
    ],
    "regression": [
        ["ridge"],
        ["random_forest", "gradient_boosting"],
        ["ridge", "random_forest", "gradient_boosting"],
    ],
}


def node_ingest(state: PipelineState) -> PipelineState:
    resp = tools.call_ingest(IngestRequest(
        run_id=state["run_id"],
        source_url=state.get("source_url"),
        raw_csv_key=state.get("raw_csv_key"),
        target_column=state["target_column"],
    ))
    return {
        "raw_data_key": resp.raw_data_key,
        "task_type": resp.task_type,
        "columns": resp.columns,
        "log": state.get("log", []) + [f"Ingested {resp.n_rows}x{resp.n_cols}, task_type={resp.task_type}"],
    }


def node_eda(state: PipelineState) -> PipelineState:
    resp = tools.call_eda(EDARequest(
        run_id=state["run_id"],
        raw_data_key=state["raw_data_key"],
        target_column=state["target_column"],
    ))
    return {
        "eda_report_key": resp.eda_report_key,
        "eda_warnings": resp.warnings,
        "high_cardinality_cols": resp.high_cardinality_cols,
        "log": state.get("log", []) + [f"EDA: {len(resp.warnings)} warning(s)"],
    }


def node_feature_engineering(state: PipelineState) -> PipelineState:
    resp = tools.call_feature_engineering(FeatureEngineeringRequest(
        run_id=state["run_id"],
        raw_data_key=state["raw_data_key"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        drop_columns=state.get("high_cardinality_cols", []),
    ))
    retries = state.get("retries", 0)
    model_sets = RETRY_MODEL_SETS[state["task_type"]]
    candidate_models = model_sets[min(retries, len(model_sets) - 1)]
    return {
        "features_key": resp.features_key,
        "feature_names": resp.feature_names,
        "candidate_models": candidate_models,
        "log": state.get("log", []) + [f"Features built: {len(resp.feature_names)} columns, "
                                        f"trying models {candidate_models}"],
    }


def node_train(state: PipelineState) -> PipelineState:
    resp = tools.call_train(TrainRequest(
        run_id=state["run_id"],
        features_key=state["features_key"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        candidate_models=state.get("candidate_models", []),
    ))
    return {
        "train_results": resp.results,
        "log": state.get("log", []) + [f"Trained {len(resp.results)} model(s)"],
    }


def node_evaluate(state: PipelineState) -> PipelineState:
    resp = tools.call_evaluate(EvaluateRequest(
        run_id=state["run_id"],
        results=state["train_results"],
        task_type=state["task_type"],
    ))
    return {
        "best_model_name": resp.best_model_name,
        "best_model_key": resp.best_model_key,
        "leaderboard": resp.leaderboard,
        "passed_quality_bar": resp.passed_quality_bar,
        "eval_reason": resp.reason,
        "log": state.get("log", []) + [f"Evaluated: best={resp.best_model_name} "
                                        f"passed={resp.passed_quality_bar} ({resp.reason})"],
    }


def node_report(state: PipelineState) -> PipelineState:
    resp = tools.call_report(ReportRequest(
        run_id=state["run_id"],
        eda_report_key=state["eda_report_key"],
        leaderboard=state["leaderboard"],
        best_model_name=state["best_model_name"],
        task_type=state["task_type"],
        narrative_notes=state.get("eval_reason"),
    ))
    return {
        "report_key": resp.report_key,
        "log": state.get("log", []) + ["Report generated"],
    }


def node_bump_retry(state: PipelineState) -> PipelineState:
    return {"retries": state.get("retries", 0) + 1}


def route_after_evaluate(state: PipelineState) -> str:
    if state.get("passed_quality_bar"):
        return "report"
    if state.get("retries", 0) < state.get("max_retries", 2):
        return "retry"
    # Out of retries: ship the best model we found anyway, with an honest report.
    return "report"


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", node_ingest)
    g.add_node("eda", node_eda)
    g.add_node("feature_engineering", node_feature_engineering)
    g.add_node("train", node_train)
    g.add_node("evaluate", node_evaluate)
    g.add_node("bump_retry", node_bump_retry)
    g.add_node("report", node_report)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "eda")
    g.add_edge("eda", "feature_engineering")
    g.add_edge("feature_engineering", "train")
    g.add_edge("train", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"report": "report", "retry": "bump_retry"},
    )
    g.add_edge("bump_retry", "feature_engineering")
    g.add_edge("report", END)

    return g.compile()


GRAPH = build_graph()
