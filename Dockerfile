FROM varshinicb99/academicos-backend:v1 AS olddata

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[api,mobile-scan]"
COPY config/ ./config/
COPY --from=olddata /app/academicos-data ./academicos-data
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "python -m academicos.cli serve --host 0.0.0.0 --port ${PORT}"]
