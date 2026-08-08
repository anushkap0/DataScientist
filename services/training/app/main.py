import logging

from fastapi import FastAPI, HTTPException
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from shared.schemas import ModelResult, TrainRequest, TrainResponse
from shared.storage import get_dataframe, put_pickle, run_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("training")

app = FastAPI(title="Training Service")

from shared.metrics import instrument
instrument(app, service_name="training")

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
}
REGRESSORS = {
    "ridge": lambda: Ridge(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        df = get_dataframe(req.features_key)
    except Exception as e:
        raise HTTPException(404, f"Could not load features_key: {e}")

    if req.target_column not in df.columns:
        raise HTTPException(422, f"target_column '{req.target_column}' not in features")

    y = df[req.target_column]
    X = df.drop(columns=[req.target_column])

    stratify = y if req.task_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    registry = CLASSIFIERS if req.task_type == "classification" else REGRESSORS
    model_names = req.candidate_models or list(registry.keys())
    unknown = [m for m in model_names if m not in registry]
    if unknown:
        raise HTTPException(422, f"Unknown model(s) {unknown} for task_type={req.task_type}. "
                                  f"Available: {list(registry.keys())}")

    results: list[ModelResult] = []
    for name in model_names:
        model = registry[name]()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if req.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
            }
            if y.nunique() == 2 and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
                except Exception:
                    pass
        else:
            metrics = {
                "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
                "mae": float(mean_absolute_error(y_test, preds)),
                "r2": float(r2_score(y_test, preds)),
            }

        model_key = run_key(req.run_id, f"model_{name}.pkl")
        put_pickle(model_key, model)

        results.append(ModelResult(model_name=name, model_key=model_key, metrics=metrics))
        log.info("Trained run=%s model=%s metrics=%s", req.run_id, name, metrics)

    return TrainResponse(run_id=req.run_id, results=results)
