from typing import Literal, Optional, TypedDict

from shared.schemas import ModelResult


class PipelineState(TypedDict, total=False):
    run_id: str
    source_url: Optional[str]
    raw_csv_key: Optional[str]
    target_column: str

    # ingestion
    raw_data_key: str
    task_type: Literal["classification", "regression"]
    columns: list[str]

    # eda
    eda_report_key: str
    eda_warnings: list[str]
    high_cardinality_cols: list[str]

    # feature engineering
    features_key: str
    feature_names: list[str]

    # training / evaluation
    candidate_models: list[str]
    train_results: list[ModelResult]
    best_model_name: str
    best_model_key: str
    leaderboard: list[ModelResult]
    passed_quality_bar: bool
    eval_reason: str

    # reporting
    report_key: str

    # control flow
    retries: int
    max_retries: int
    log: list[str]
    error: Optional[str]
