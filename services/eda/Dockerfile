# Build context MUST be the repo root (see scripts/build_images.sh) so this
# can pull in shared/ alongside the service's own code.
FROM python:3.11-slim

WORKDIR /app

COPY services/eda/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY services/eda/app ./app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
