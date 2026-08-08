#!/bin/bash
# Builds every service image with the repo root as build context (required
# so each Dockerfile can COPY the shared/ package). Usage:
#   REGISTRY=ghcr.io/yourorg TAG=v1 ./scripts/build_images.sh
set -euo pipefail

REGISTRY="${REGISTRY:-localhost:5000}"
TAG="${TAG:-latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVICES=(ingestion eda feature_engineering training evaluation reporting orchestrator)

for svc in "${SERVICES[@]}"; do
  name="ads-${svc//_/-}"
  echo "Building ${REGISTRY}/${name}:${TAG} ..."
  docker build -f "${ROOT_DIR}/services/${svc}/Dockerfile" -t "${REGISTRY}/${name}:${TAG}" "${ROOT_DIR}"
  docker push "${REGISTRY}/${name}:${TAG}"
done

echo "Done. Update k8s/*.yaml image fields to ${REGISTRY}/ads-<service>:${TAG}, or run scripts/deploy.sh which does this for you."
