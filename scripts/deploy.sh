#!/bin/bash
# Substitutes the REGISTRY/TAG placeholder in the k8s manifests and applies
# them in order. Run scripts/build_images.sh first.
#   REGISTRY=ghcr.io/yourorg TAG=v1 ./scripts/deploy.sh
set -euo pipefail

REGISTRY="${REGISTRY:-localhost:5000}"
TAG="${TAG:-latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cp "${ROOT_DIR}"/k8s/*.yaml "${TMP_DIR}/"
sed -i.bak "s#REGISTRY/\([a-z-]*\):latest#${REGISTRY}/\1:${TAG}#g" "${TMP_DIR}"/*.yaml
rm -f "${TMP_DIR}"/*.bak

echo "Applying manifests from ${TMP_DIR} ..."
kubectl apply -f "${TMP_DIR}/00-namespace-config.yaml"

echo
echo ">>> Before continuing, make sure these secrets exist in namespace autonomous-ds:"
echo "    kubectl create secret generic minio-credentials -n autonomous-ds \\"
echo "      --from-literal=access-key=minioadmin --from-literal=secret-key=<CHANGE_ME>"
echo "    kubectl create secret generic anthropic-credentials -n autonomous-ds \\"
echo "      --from-literal=api-key=<YOUR_ANTHROPIC_API_KEY>"
echo "    # Only needed if you set REPORT_LLM_PROVIDER=huggingface in k8s/00-namespace-config.yaml:"
echo "    kubectl create secret generic hf-credentials -n autonomous-ds \\"
echo "      --from-literal=api-token=<YOUR_HUGGING_FACE_TOKEN>"
read -p "Press Enter once secrets are created to continue applying the rest..."

for f in "${TMP_DIR}"/0[1-9]-*.yaml; do
  kubectl apply -f "$f"
done

echo "Waiting for orchestrator rollout..."
kubectl -n autonomous-ds rollout status deployment/orchestrator

echo "Done. Port-forward to try it locally:"
echo "  kubectl -n autonomous-ds port-forward svc/orchestrator-svc 8000:8000"
