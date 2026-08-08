# Build context MUST be the repo root (see scripts/build_images.sh) so this
# can pull in shared/ alongside the service's own code.
FROM python:3.11-slim

WORKDIR /app

COPY services/feature_engineering/requirements.txt ./requirements.txt
# torch is a huge dependency mainly because of its default CUDA build.
# This is a CPU-only service (no GPU in these pods), so install the
# CPU-only wheel explicitly first, then let the rest of requirements.txt
# install normally (pip will see torch already satisfied and skip it).
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY services/feature_engineering/app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
