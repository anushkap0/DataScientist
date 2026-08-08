# Autonomous Data Scientist

An agentic pipeline that takes a raw CSV + a target column and autonomously
ingests, profiles, cleans, trains, evaluates, and writes up a modeling
report — no human in the loop. Orchestrated by **LangGraph**, built as
independently scalable **FastAPI microservices**, deployed on **Kubernetes**.

## Architecture

```
                        ┌──────────────────────┐
   POST /run  ────────▶ │     orchestrator      │  (LangGraph agent graph)
                        │  1 replica, in-mem    │
                        └──────────┬────────────┘
                                   │ REST calls, one per pipeline stage
        ┌───────────┬─────────────┼─────────────┬────────────┬───────────┐
        ▼           ▼             ▼             ▼            ▼           ▼
   ingestion       eda      feature-eng     training     evaluation  reporting
   (N pods)      (N pods)     (N pods)      (N pods)      (N pods)   (N pods, LLM)
        └───────────┴─────────────┴─────────────┴────────────┴───────────┘
                                   │
                                   ▼
                          MinIO (S3-compatible)
                     shared artifact store, keyed by run_id
```

**Why this shape:** every service is stateless — it reads its input
artifact from MinIO, does one job, writes its output artifact back, and
returns. That's what makes them independently horizontally scalable (each
has its own `Deployment` + `HorizontalPodAutoscaler` in `k8s/`): if training
is the bottleneck, its HPA scales it up to 8 pods while `evaluation` stays
at 1. Only the orchestrator is a singleton, because it holds an in-memory
run registry (see "Scaling further" below).

### The agent graph

```
ingest → eda → feature_engineering → train → evaluate ──┬──▶ report → END
                        ▲                                │
                        └──────── retry (bad metrics) ────┘
```

This is what makes it "autonomous" rather than a fixed script: the
`evaluate` node applies a quality bar per task type (e.g. F1 ≥ 0.55 for
classification). If the best model misses it and retries remain, the graph
loops back to `feature_engineering`, which escalates to a richer candidate
model set on each pass (`services/orchestrator/app/graph.py`,
`RETRY_MODEL_SETS`). If retries run out, it ships the best model found
anyway with an honest report rather than hanging forever.

The `reporting` service is the one LLM-backed step: it calls the Anthropic
API to turn the EDA + leaderboard JSON into a written report, with a
deterministic fallback if the call fails so the pipeline never dead-ends.

## Services

| Service | Endpoint | Job |
|---|---|---|
| `orchestrator` | `POST /run`, `GET /status/{run_id}`, `GET /artifacts/{run_id}/{name}` | Runs the LangGraph agent |
| `ingestion` | `POST /ingest` | Loads CSV (URL or object key), validates, infers task type |
| `eda` | `POST /analyze` | Profiles missingness, correlations, cardinality, warnings |
| `feature_engineering` | `POST /transform` | Imputes, scales, encodes |
| `training` | `POST /train` | Trains candidate sklearn models, computes metrics |
| `evaluation` | `POST /evaluate` | Picks best model, applies quality gate |
| `reporting` | `POST /report` | LLM-written markdown report |

Request/response contracts for all of the above live in one place:
`shared/schemas.py`.

## Run it locally (docker-compose)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

Then kick off a run:

```bash
curl -X POST localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
        "source_url": "https://raw.githubusercontent.com/<you>/<repo>/main/examples/sample_churn.csv",
        "target_column": "churned",
        "max_retries": 2
      }'
# => {"run_id": "...", "status": "queued"}

curl localhost:8000/status/<run_id>
curl localhost:8000/artifacts/<run_id>/report.md
```

A tiny sample dataset is at `examples/sample_churn.csv` — upload it
somewhere reachable by URL, or `PUT` it into the `ads-artifacts` MinIO
bucket (console at `localhost:9001`, minioadmin/minioadmin) and pass
`"raw_csv_key": "your-key.csv"` instead of `source_url`.

## Deploy to Kubernetes

```bash
export REGISTRY=ghcr.io/yourorg
export TAG=v1
./scripts/build_images.sh   # builds + pushes all 7 images
./scripts/deploy.sh         # applies k8s/, prompts you to create secrets first
kubectl -n autonomous-ds port-forward svc/orchestrator-svc 8000:8000
```

Manifests are numbered so `kubectl apply -f k8s/` (or the deploy script)
applies namespace → config → MinIO → services in a sane order. Each
compute service (`ingestion`, `eda`, `feature_engineering`, `training`,
`evaluation`, `reporting`) ships with its own `HorizontalPodAutoscaler`
targeting 70% CPU. Swap the CPU metric for a custom queue-depth metric if
you front the services with a queue later (see below).

## Scaling further

- **Orchestrator statefulness**: `RUNS` is an in-memory dict
  (`services/orchestrator/app/main.py`). Fine for 1 replica; to scale
  orchestrator horizontally, move that registry to Redis (run id → status)
  so any replica can answer `/status/{run_id}`.
- **Async fan-out**: today the orchestrator calls each service
  synchronously over REST per run. For higher throughput, put a queue
  (e.g. NATS/RabbitMQ) between `train` and its candidate models so each
  candidate model trains as its own message — that's how you'd get
  `training` to scale per-model rather than per-run.
- **Smarter agent decisions**: `RETRY_MODEL_SETS` in `graph.py` is a
  deterministic escalation ladder. Swap it for an LLM call that reads the
  EDA report and picks a model family and hyperparameter search space —
  that's the natural next "agent" to add.
- **Real datasets**: `training`'s in-memory `train_test_split` won't scale
  past what fits in one pod's memory. For big data, swap pandas for a
  Ray/Dask/Spark-backed training service behind the same REST contract —
  nothing else in the graph needs to change, since `shared/schemas.py` is
  the only coupling between services.

## Observability & getting real latency/success-rate numbers

Every service exposes Prometheus metrics at `GET /metrics`
(`shared/metrics.py`, added via one `instrument(app, service_name=...)`
call per service):

- `http_request_latency_seconds{service, method, path}` — histogram, gives you p50/p95/p99 per endpoint
- `http_requests_total{service, method, path, status}` — counter, gives you success rate (`status=~"2.."` vs the rest)
- `http_requests_in_flight{service}` — concurrency gauge
- Orchestrator only: `pipeline_runs_total{outcome}`, `pipeline_run_duration_seconds`, `pipeline_retries` — the business-level numbers (end-to-end pipeline success rate, run duration), separate from raw API latency

**Locally:** `docker compose up` also starts Prometheus at `localhost:9090`
(`observability/prometheus.yml` scrapes all 7 services). Query e.g.
`histogram_quantile(0.95, rate(http_request_latency_seconds_bucket[5m]))`.

**On Kubernetes:** `k8s/09-prometheus.yaml` deploys an in-cluster
Prometheus that auto-discovers any pod annotated
`prometheus.io/scrape: "true"` (already set on every Deployment). At
production scale, swap it for the `kube-prometheus-stack` Helm chart
(adds Grafana + alerting) — the app side doesn't change, since services
just expose `/metrics` regardless of who scrapes it.

**To get an actual number to put on a resume**, don't estimate one — run
`scripts/loadtest.py` (Locust) against a deployed stack and report what it
prints:

```bash
pip install locust
locust -f scripts/loadtest.py --host http://localhost:8000 \
  --users 20 --spawn-rate 2 --run-time 5m --headless --csv results/run1
```

This separates two different things people conflate: **API latency**
(`/health`, `/run` accept-time — should be tens of milliseconds since
`/run` just enqueues a background task) vs. **pipeline completion time**
(`pipeline_run_duration_seconds` — seconds to minutes, dominated by model
training). Report them separately, and report the HPA behavior too if you
load-test hard enough to trigger a scale-up (`kubectl get hpa -n
autonomous-ds -w` while the test runs) — "autoscaled from 2→6 pods under
load, keeping p95 latency under Xms" is a more credible, specific claim
than a single unqualified percentage.

## Using Hugging Face

Hugging Face plugs in at two points:

**1. Free-text feature embeddings (feature_engineering service).** If a
categorical column looks like free text (average ≥4 words per value —
reviews, comments, descriptions) rather than a short label, it's embedded
with a sentence-transformer (`HF_EMBEDDING_MODEL`, default
`sentence-transformers/all-MiniLM-L6-v2`) instead of label-encoded, then
reduced to `TEXT_EMBEDDING_COMPONENTS` dims (default 16) via SVD so it
doesn't dwarf the rest of the feature space. This model is public — no
token needed. It's on by default; if the model can't load (no network,
package missing), that column silently falls back to label encoding
rather than failing the run — check the service logs if you expect
embeddings and don't see `_emb_` columns in the feature set.

Note: this adds `sentence-transformers` + `torch` to that one service's
image (~1-1.5GB with the CPU-only torch wheel the Dockerfile installs
explicitly — see `services/feature_engineering/Dockerfile`).

**2. Report generation (reporting service).** Set
`REPORT_LLM_PROVIDER=huggingface` (default is `anthropic`) plus
`HF_API_TOKEN` and `HF_MODEL` (default
`meta-llama/Meta-Llama-3-8B-Instruct`) to write the final report with a
model served over the Hugging Face Inference API instead of Claude. Same
fallback behavior either way — if the call fails, you get the
deterministic report, not a broken pipeline. Swap the provider by
changing one env var; `services/reporting/app/llm.py` is the whole
abstraction (`services/reporting/app/main.py` doesn't know which provider
it's talking to).

Locally, set these in `.env` (see `.env.example`); on Kubernetes, they're
in `k8s/00-namespace-config.yaml`'s ConfigMap plus the optional
`hf-credentials` secret (`scripts/deploy.sh` prints the `kubectl create
secret` command for it).

## Repo layout

```
shared/                  # schemas + MinIO client, copied into every image
services/<name>/app/     # one FastAPI app per service
services/<name>/Dockerfile
services/<name>/requirements.txt
k8s/                      # numbered manifests, apply in order (09 = Prometheus)
scripts/                  # build_images.sh, deploy.sh, loadtest.py (Locust)
observability/prometheus.yml   # scrape config for docker-compose
docker-compose.yml         # local dev stack (includes Prometheus)
examples/sample_churn.csv
```
