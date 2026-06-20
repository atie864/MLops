from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class ListingFeatures(BaseModel):
    room_type: str
    property_type: str
    neighbourhood_name: str
    accommodates: int
    minimum_nights: int
    maximum_nights: int
    host_listing_count: int
    available_days_last_90d: int
    available_days_last_30d: int
    instant_bookable: bool
    is_superhost: bool
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    bathrooms: Optional[float] = None
    listing_price: Optional[float] = None
    total_reviews_before_cutoff: Optional[float] = None
    unique_reviewers_before_cutoff: Optional[float] = None
    avg_comment_len_before_cutoff: Optional[float] = None
    max_comment_len_before_cutoff: Optional[float] = None
    days_since_last_review: Optional[float] = None
    available_rate_last_90d: Optional[float] = None
    avg_minimum_nights_calendar_last_90d: Optional[float] = None
    avg_maximum_nights_calendar_last_90d: Optional[float] = None
    available_rate_last_30d: Optional[float] = None
    avg_minimum_nights_calendar_last_30d: Optional[float] = None
    avg_maximum_nights_calendar_last_30d: Optional[float] = None

class PredictionResponse(BaseModel):
    listing_id: Optional[int] = None
    prediction: int
    probability_high_demand: float
    model_run_id: str
