import pandas as pd 
from typing import Iterable

def build_neighbourhood_summary(
        listings: pd.DataFrame,
        segments: pd.DataFrame
):
    summary=(listings.groupby("neighbourhood")
             .agg(
                 num_listings=("listing_id","count"),
                 avg_price=("price","mean"),
                 median_price=("price","median"),
                 avg_minimum_nights=("minimum_nights","mean"),
                 availability_365_avg=("availability_365","mean"),
                 total_reviews=("number_of_reviews","sum")

             ))
    summary.head()
    summary["reviews_per_listing"]=(
        summary["total_reviews"]/summary["num_listings"]
    )
    summary=summary.merge(
        segments[["neighbourhood","tourism_segment","priority_level"]],
        on="neighbourhood",how="left"
    )
    summary["tourism_segment"]=(summary["tourism_segment"].fillna("unknown"))
    
    output_columns = [
        "neighbourhood",
        "num_listings",
        "avg_price",
        "median_price",
        "avg_minimum_nights",
        "availability_365_avg",
        "total_reviews",
        "reviews_per_listing",
        "tourism_segment",
        "priority_level",
    ]
    return summary[output_columns]