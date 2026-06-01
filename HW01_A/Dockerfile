FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir -e .
CMD [ "airbnb-ops", "run" ]

