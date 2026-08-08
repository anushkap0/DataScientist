"""
Drop-in request metrics for every service. Adds:
  - http_requests_total{service, method, path, status}      (Counter)
  - http_request_latency_seconds{service, method, path}     (Histogram)
  - http_requests_in_flight{service}                         (Gauge)
and exposes them on GET /metrics in Prometheus text format.

Usage in a service's main.py:

    from shared.metrics import instrument
    app = FastAPI(...)
    instrument(app, service_name="eda")

That's the entire integration — no per-endpoint code needed. Histogram
buckets are tuned for sub-second-to-multi-second ML/API workloads.
"""
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["service", "method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "Request latency in seconds",
    ["service", "method", "path"], buckets=_BUCKETS,
)
IN_FLIGHT = Gauge(
    "http_requests_in_flight", "Requests currently being processed",
    ["service"],
)


def instrument(app: FastAPI, service_name: str) -> None:
    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        # Keep cardinality bounded: use the route template ("/status/{run_id}"),
        # not the raw path, so per-run_id paths don't create a new label value
        # for every single request.
        path = request.url.path
        method = request.method
        IN_FLIGHT.labels(service=service_name).inc()
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            template = route.path if route is not None else path
            REQUEST_LATENCY.labels(service=service_name, method=method, path=template).observe(elapsed)
            REQUEST_COUNT.labels(service=service_name, method=method, path=template, status=status).inc()
            IN_FLIGHT.labels(service=service_name).dec()
        return response

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
