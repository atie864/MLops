from __future__ import annotations
import os
from contextlib import asynccontextmanager
import mlflow.sklearn
from fastapi import FastAPI
from airbnb_serving.predictor import predict_batch, predict_single
from airbnb_serving.schema import ListingFeatures, PredictionResponse
from typing import List
from dotenv import load_dotenv
load_dotenv()
_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
    run_id = os.environ["MODEL_RUN_ID"]
    mlflow.set_tracking_uri(tracking_uri)
    _state["model"] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    _state["run_id"] = run_id
    yield
    _state.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": _state.get("run_id", "")}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: ListingFeatures) -> PredictionResponse:
    return predict_single(features, _state["model"], _state["run_id"])

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch_endpoint(features_list: List[ListingFeatures]) -> List[PredictionResponse]:
    return predict_batch(features_list, _state["model"], _state["run_id"])
