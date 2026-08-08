"""
Load test for the orchestrator's /run endpoint. This is what turns
"has metrics" into an actual number you can cite: run this against your
deployed stack and report what it produces — don't estimate.

Usage:
    pip install locust
    locust -f scripts/loadtest.py --host http://localhost:8000 \
           --users 20 --spawn-rate 2 --run-time 5m --headless \
           --csv results/run1

Locust's own report (or the printed summary) gives you:
    - requests/sec (throughput)
    - p50 / p95 / p99 latency
    - failure %  (your "success rate")

Point SOURCE_URL at a small, fast, reachable CSV — the sample in
examples/sample_churn.csv pushed to a gist works well. Using a large
dataset will make "latency" mostly mean "model training time", which is
a different number than API latency — keep them separate when you report
results.
"""
import os
import time

from locust import HttpUser, between, task

SOURCE_URL = os.getenv("LOADTEST_SOURCE_URL", "https://example.com/sample_churn.csv")
TARGET_COLUMN = os.getenv("LOADTEST_TARGET_COLUMN", "churned")


class OrchestratorUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        # Measures pure API latency (routing + FastAPI overhead), separate
        # from pipeline execution time. This is the number to quote for
        # "p95 API latency".
        self.client.get("/health", name="/health")

    @task(1)
    def submit_run(self):
        # Measures time to accept a job (should stay fast since the graph
        # runs in a background task) — NOT total pipeline completion time.
        with self.client.post(
            "/run",
            json={"source_url": SOURCE_URL, "target_column": TARGET_COLUMN, "max_retries": 1},
            name="/run",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code}")
                return
            run_id = resp.json().get("run_id")
            if not run_id:
                resp.failure("no run_id in response")

    @task(1)
    def poll_status(self):
        # Optional: pick a recent run_id and poll it. Left as a manual
        # extension point — wire this to a shared list of run_ids captured
        # from submit_run if you want end-to-end pipeline latency too.
        pass
