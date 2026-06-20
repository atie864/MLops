# Stage 1 - builder
FROM python:3.8-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir build wheel
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install --no-deps -e .

# Stage 2 - runtime
FROM python:3.8-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "airbnb_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
