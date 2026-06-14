from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    # TODO 1: reject unknown fields and forbidden leakage fields.
    forbidden_present = [f for f in config.FORBIDDEN_FIELDS if f in df.columns]
    if forbidden_present:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Request contains forbidden leakage/audit fields.",
                "forbidden_fields": forbidden_present,
            },
        )


    # TODO 2: check missing fields against config.EXPECTED_FEATURE_COLUMNS.
    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "TODO: missing feature handling", "missing_fields": missing_cols},
        )
    df = df.rename(columns={"host_is_superhost": "is_superhost"})
 
    # Build the model's actual feature list (with the renamed column)
    model_feature_columns = [
        "is_superhost" if c == "host_is_superhost" else c
        for c in config.EXPECTED_FEATURE_COLUMNS
    ]

    return df[model_feature_columns]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """TODO: Run model prediction and return API responses."""
    X = records_to_dataframe(records)

    # TODO:
    # - if model has predict_proba, use positive-class probability.
    # - apply config.PREDICTION_THRESHOLD.

    if hasattr(model, "predict_proba"):
        proba_matrix = model.predict_proba(X)
        probabilities = proba_matrix[:, 1].tolist()
        predictions = [
            1 if p >= config.PREDICTION_THRESHOLD else 0
            for p in probabilities
        ]
    else:
        predictions = model.predict(X).tolist()
        probabilities = [None] * len(predictions)

    # - return one PredictionResponse per record.
    # Temporary placeholder so the endpoint shape is clear:

    responses = []
    for pred, prob in zip(predictions, probabilities):
        responses.append(
            PredictionResponse(
                prediction=int(pred),
                prediction_label=(
                    config.POSITIVE_LABEL if pred == 1 else config.NEGATIVE_LABEL
                ),
                probability=round(float(prob), 6) if prob is not None else None,
                threshold=config.PREDICTION_THRESHOLD,
            )
        )

    return responses

