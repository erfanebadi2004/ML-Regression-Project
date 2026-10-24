ARG PY_VERSION=3.12-slim

FROM python:${PY_VERSION}

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN for i in 1 2 3; do \
        apt-get update && apt-get install -y --no-install_recommends libgomp1 && rm -rf /var/lib/apt/lists/* && break || \
        sleep 5; \
    done

EXPOSE 8000

CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "8000"]