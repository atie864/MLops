from __future__ import annotations
import pandas as pd
from airbnb_serving.schema import ListingFeatures, PredictionResponse
from typing import List

FEATURE_COLS = [
    'room_type', 'property_type', 'neighbourhood_name',
    'accommodates', 'bedrooms', 'beds', 'bathrooms', 'listing_price',
    'minimum_nights', 'maximum_nights', 'instant_bookable', 'is_superhost',
    'host_listing_count', 'total_reviews_before_cutoff', 'unique_reviewers_before_cutoff',
    'avg_comment_len_before_cutoff', 'max_comment_len_before_cutoff',
    'days_since_last_review', 'available_days_last_90d', 'available_rate_last_90d',
    'avg_minimum_nights_calendar_last_90d', 'avg_maximum_nights_calendar_last_90d',
    'available_days_last_30d', 'available_rate_last_30d',
    'avg_minimum_nights_calendar_last_30d', 'avg_maximum_nights_calendar_last_30d',
]

def predict_single(features: ListingFeatures, model, run_id: str) -> PredictionResponse:
    df = pd.DataFrame([features.model_dump()])[FEATURE_COLS]
    prediction = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    return PredictionResponse(prediction=prediction, probability_high_demand=proba, model_run_id=run_id)

def predict_batch(features_list: List[ListingFeatures], model, run_id: str) -> List[PredictionResponse]:
    df = pd.DataFrame([f.model_dump() for f in features_list])[FEATURE_COLS]
    predictions = model.predict(df)
    probas = model.predict_proba(df)[:, 1]
    return [
        PredictionResponse(prediction=int(p), probability_high_demand=float(prob), model_run_id=run_id)
        for p, prob in zip(predictions, probas)
    ]
