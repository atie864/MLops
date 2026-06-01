import pandas as pd

REQUIRED_OUTPUT_COLUMNS = {
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
}
PII_COLUMNS = {
    "host_name",
    "host_id",
}
def validate_summary(summary:pd.DataFrame):

    if summary.empty:
        raise ValueError("Summary output is empty.")
    
    miss_colmns=REQUIRED_OUTPUT_COLUMNS-set(summary.columns)
    if miss_colmns:
        raise ValueError(
            f"Missing required output columns: {miss_colmns}"
        )
    
    found_pii=PII_COLUMNS.intersection(summary.columns)
    if found_pii:
        raise ValueError(f"PII columns found in output:{found_pii}")
    
    if summary["neighbourhood"].isna().any():
        raise ValueError("found null values in neighbourhood column")
    
    if (summary["num_listings"]<=0).any():
        raise ValueError("num_listings must be greater than 0.")
    
    if (summary["avg_price"]<0).any():
        raise ValueError("avg_price must be greater than 0.")
    
    invalid_availblty=((summary["availability_365_avg"]<0) | (summary["availability_365_avg"]>355))
    if invalid_availblty.any():
        raise ValueError("availability_365_avg must be between 0 and 365.")
       