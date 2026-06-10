import os
import json
import re
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from typing import Optional
def read_sql(query: str, params:Optional[dict] =None ) -> pd.DataFrame:
    """Run a SQL query through SQLAlchemy and return a Pandas DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)

# -----------------------------
# ETL Configuration
# -----------------------------
DATASET_VERSION = "v1_student"

ENTITY_COLUMN = "listing_id"

PAST_WINDOW_DAYS = 90
FUTURE_WINDOW_DAYS = 30
HIGH_DEMAND_AVAILABLE_RATE_THRESHOLD = 0.30

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("FEATURE_DIR:", FEATURE_DIR)

# -----------------------------
# Database Connection
# -----------------------------

# Clear old environment variables that may point to the wrong database.
# for key in ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]:
#     os.environ.pop(key, None)
load_dotenv()
DB_HOST = os.getenv("PGHOST", "")
DB_PORT = int(os.getenv("PGPORT", ""))
DB_NAME = os.getenv("PGDATABASE", "")
DB_USER = os.getenv("PGUSER", "")
DB_PASSWORD = os.getenv("PGPASSWORD", "")


if DB_USER == "student_your_username" or DB_PASSWORD == "student_your_password":
    raise ValueError("Replace DB_USER and DB_PASSWORD with your assigned database credentials.")

print("Connecting to:")
print("HOST:", DB_HOST)
print("PORT:", DB_PORT)
print("DB:", DB_NAME)
print("USER:", DB_USER)

db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    query={"sslmode": "disable"},
)

engine = create_engine(db_url)


with engine.connect() as conn:
    connection_check = conn.execute(
        text("""
        SELECT
            current_database() AS database,
            current_user AS user_name,
            inet_server_addr() AS server_ip,
            inet_server_port() AS server_port,
            now() AS checked_at;
        """)
    ).mappings().first()

print(dict(connection_check))

tables_df = read_sql("""
SELECT
    table_schema,
    table_name,
    table_type
FROM information_schema.tables
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name;
""")

print(tables_df)

columns_df = read_sql("""
SELECT
    table_schema,
    table_name,
    ordinal_position,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'core'
ORDER BY table_schema, table_name, ordinal_position;
""")

print(columns_df)

row_counts_df = read_sql("""
SELECT 'core.calendar_day' AS table_name, COUNT(*) AS row_count FROM core.calendar_day
UNION ALL
SELECT 'core.host' AS table_name, COUNT(*) AS row_count FROM core.host
UNION ALL
SELECT 'core.listing' AS table_name, COUNT(*) AS row_count FROM core.listing
UNION ALL
SELECT 'core.neighbourhood' AS table_name, COUNT(*) AS row_count FROM core.neighbourhood
UNION ALL
SELECT 'core.review' AS table_name, COUNT(*) AS row_count FROM core.review
ORDER BY table_name;
""")

print(row_counts_df)

calendar_quality_df = read_sql("""
SELECT
    COUNT(*) AS n_rows,
    COUNT(*) FILTER (WHERE price IS NULL) AS null_price,
    COUNT(*) FILTER (WHERE adjusted_price IS NULL) AS null_adjusted_price,
    COUNT(*) FILTER (WHERE available IS NULL) AS null_available,
    MIN(date) AS min_calendar_date,
    MAX(date) AS max_calendar_date
FROM core.calendar_day;
""")

review_quality_df = read_sql("""
SELECT
    COUNT(*) AS n_rows,
    COUNT(*) FILTER (WHERE comment_len IS NULL) AS null_comment_len,
    MIN(review_date) AS min_review_date,
    MAX(review_date) AS max_review_date
FROM core.review;
""")

print(calendar_quality_df)
print(review_quality_df)

# Inspect small samples.
# Keep LIMIT small. Do not pull full raw calendar/review tables into Pandas.

for table_name in ["listing", "host", "neighbourhood", "review", "calendar_day"]:
    print(f"\n===== core.{table_name} =====")
    print(read_sql(f"SELECT * FROM core.{table_name} LIMIT 10;"))

range_df = read_sql("""
SELECT
    (SELECT MIN(date) FROM core.calendar_day) AS calendar_min_date,
    (SELECT MAX(date) FROM core.calendar_day) AS calendar_max_date,
    (SELECT MIN(review_date) FROM core.review) AS review_min_date,
    (SELECT MAX(review_date) FROM core.review) AS review_max_date;
""")

print(range_df)

calendar_min_date = pd.to_datetime(range_df.loc[0, "calendar_min_date"]).date()
calendar_max_date = pd.to_datetime(range_df.loc[0, "calendar_max_date"]).date()

# part 5 ..............

earliest_cutoff_allowed_by_calendar = calendar_min_date + timedelta(days=PAST_WINDOW_DAYS)
latest_cutoff_allowed_by_calendar = calendar_max_date - timedelta(FUTURE_WINDOW_DAYS)

cutoff_date = latest_cutoff_allowed_by_calendar
history_start_date = cutoff_date - timedelta(days=PAST_WINDOW_DAYS)
label_end_date = cutoff_date + timedelta(FUTURE_WINDOW_DAYS)

# TODO: print all dates and assert that the windows are valid.
print("calendar_min_date       :", calendar_min_date)
print("calendar_max_date       :", calendar_max_date)
print("earliest_cutoff_allowed :", earliest_cutoff_allowed_by_calendar)
print("latest_cutoff_allowed   :", latest_cutoff_allowed_by_calendar)
print("cutoff_date             :", cutoff_date)
print("history_start_date      :", history_start_date)
print("label_end_date          :", label_end_date)

assert earliest_cutoff_allowed_by_calendar <= cutoff_date <= latest_cutoff_allowed_by_calendar, \
    "cutoff_date is outside the valid range"
assert history_start_date >= calendar_min_date, \
    "history_start_date is before the earliest calendar data"
assert label_end_date <= calendar_max_date, \
    "label_end_date is after the latest calendar data"
assert (cutoff_date - history_start_date).days == PAST_WINDOW_DAYS, \
    "past window size mismatch"
assert (label_end_date - cutoff_date).days == FUTURE_WINDOW_DAYS, \
    "future window size mismatch"

print("\nAll date assertions passed.")
# TODO: complete the PII audit table.
# Add rows for all sensitive or identity-linking columns you find relevant.

pii_audit = pd.DataFrame([
    {
        "table": "listing",
        "column": "listing_id",
        "pii_type": "entity identifier",
        "decision": "keep as entity key only",
        "reason": "needed to define one row per listing; not a model input"
    },
    {
        "table": "listing",
        "column": "host_id",
        "pii_type": "direct identifier",
        "decision": "drop after join",
        "reason": "links directly to a host; not safe as a model feature"
    },
    {
        "table": "host",
        "column": "host_pseudo_id",
        "pii_type": "pseudonymous identifier",
        "decision": "drop",
        "reason": "pseudonymous but still an individual-level identifier; not a model feature"
    },
    {
        "table": "listing",
        "column": "license",
        "pii_type": "regulatory identifier",
        "decision": "drop",
        "reason": "may identify a specific host or property; not useful as an ML feature"
    },
    {
        "table": "review",
        "column": "review_id",
        "pii_type": "entity identifier",
        "decision": "drop",
        "reason": "row-level review identifier; only aggregates are used"
    },
    {
        "table": "review",
        "column": "reviewer_id",
        "pii_type": "direct identifier",
        "decision": "drop",
        "reason": "identifies individual reviewer; not a model input"
    },
    {
        "table": "review",
        "column": "reviewer_pseudo_id",
        "pii_type": "pseudonymous identifier",
        "decision": "drop",
        "reason": "pseudonymous but still individual-level; not a model feature"
    },
])
print(pii_audit)

# Load listing columns needed for features (exclude raw PII)
listing_df = read_sql("""
SELECT
    listing_id,
    host_id,
    neighbourhood_id,
    property_type,
    room_type,
    accommodates,
    bathrooms_text,
    bedrooms,
    beds,
    listing_price,
    minimum_nights,
    maximum_nights,
    instant_bookable,
    license
FROM core.listing;
""")

# Load host columns (keep host_id for join; drop after merging)
host_df = read_sql("""
SELECT
    host_id,
    host_pseudo_id,
    is_superhost
FROM core.host;
""")

# Load neighbourhood columns; rename 'name' -> 'neighbourhood_name'
neighbourhood_df = read_sql("""
SELECT
    neighbourhood_id,
    name AS neighbourhood_name
FROM core.neighbourhood;
""")

print("listing:", listing_df.shape)
print("host:", host_df.shape)
print("neighbourhood:", neighbourhood_df.shape)
# normalize boolean columns.
# Example target columns:
# - listing_df["instant_bookable"]
# - host_df["is_superhost"]
def to_bool(series):
    """Convert t/f strings or 0/1 integers to boolean."""
    if series.dtype == bool:
        return series
    mapping = {"t": True, "f": False, "true": True, "false": False, "1": True, "0": False}
    return series.astype(str).str.lower().str.strip().map(mapping).astype("boolean")

listing_df["instant_bookable"] = to_bool(listing_df["instant_bookable"])
host_df["is_superhost"] = to_bool(host_df["is_superhost"])


# normalize numeric listing columns.
numeric_cols_listing = [
    "accommodates",
    "bedrooms",
    "beds",
    "listing_price",
    "minimum_nights",
    "maximum_nights",
]

for col in numeric_cols_listing:
    listing_df[col] = pd.to_numeric(listing_df[col])


def parse_bathrooms(text_value):
    """
    Convert bathrooms_text into a number.

    Examples:
    - '1 bath' -> 1.0
    - '1.5 baths' -> 1.5
    - 'Half-bath' -> 0.5
    - missing/unrecognized -> NaN
    """
    if pd.isna(text_value):
        return np.nan
    text = str(text_value).strip().lower()
    if "half" in text:
        return 0.5
    # Extract the first number that appears in the string
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return np.nan


listing_df["bathrooms"] = listing_df["bathrooms_text"].apply(parse_bathrooms)

print(listing_df[["bathrooms_text", "bathrooms"]].head(10))

# create host_listing_features with one row per host_id.
host_listing_features = (
    listing_df.groupby("host_id", as_index=False)
    .agg(host_listing_count=("listing_id", "count"))
)

# merge listing, host, host_listing_features, and neighbourhood.
base_listing_features = listing_df.merge(host_df, on="host_id", how="left")

base_listing_features = base_listing_features.merge(
    host_listing_features, on="host_id", how="left"
)

base_listing_features = base_listing_features.merge(
    neighbourhood_df, on="neighbourhood_id", how="left"
)

# choose privacy-safe static feature columns.
static_feature_cols = [
    "listing_id",
    # Listing attributes
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "listing_price",
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
    # Host attributes (aggregated / non-identifying)
    "is_superhost",
    "host_listing_count",
    # Neighbourhood attributes
    "neighbourhood_name"
]

static_features = base_listing_features[static_feature_cols].copy()


static_features = base_listing_features[static_feature_cols].copy()

assert static_features["listing_id"].duplicated().sum() == 0

print(static_features.shape)
print(static_features.head())
# part 10 ..........................................
# write SQL aggregation over core.review.
# Use CAST(:cutoff_date AS date) when using the cutoff inside SQL.

review_features = read_sql(
    """
    SELECT
        listing_id,
        COUNT(*)                              AS total_reviews_before_cutoff,
        COUNT(DISTINCT reviewer_pseudo_id)    AS unique_reviewers_before_cutoff,
        AVG(comment_len)                      AS avg_comment_len_before_cutoff,
        MAX(comment_len)                      AS max_comment_len_before_cutoff,
        CAST(
            CAST(:cutoff_date AS date) - MAX(review_date)
        AS INTEGER)                           AS days_since_last_review
    FROM core.review
    WHERE review_date <= CAST(:cutoff_date AS date)
    GROUP BY listing_id;
    """,
    params={"cutoff_date": cutoff_date},
)

# convert feature columns to numeric where needed.
for col in [
    "total_reviews_before_cutoff",
    "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff",
    "max_comment_len_before_cutoff",
    "days_since_last_review",
]:
    review_features[col] = pd.to_numeric(review_features[col], errors="coerce")

assert review_features["listing_id"].duplicated().sum() == 0

print(review_features.shape)
print(review_features.head())


# write SQL aggregation over core.calendar_day.
# Use history_start_date and cutoff_date.
#
# Required output examples:
# - available_days_last_90d
# - available_rate_last_90d
# - avg_minimum_nights_calendar_last_90d
# - avg_maximum_nights_calendar_last_90d
# - available_days_last_30d
# - available_rate_last_30d
# - avg_minimum_nights_calendar_last_30d
# - avg_maximum_nights_calendar_last_30d

calendar_features_all = read_sql(
    """
    SELECT
        listing_id,
        COUNT(*)                                           AS available_days_last_90d_total,
        SUM(CASE WHEN available = 't' THEN 1 ELSE 0 END)  AS available_days_last_90d,
        ROUND(
            SUM(CASE WHEN available = 't' THEN 1.0 ELSE 0 END) / NULLIF(COUNT(*), 0),
            4
        )                                                  AS available_rate_last_90d,
        AVG(minimum_nights)                                AS avg_minimum_nights_calendar_last_90d,
        AVG(maximum_nights)                                AS avg_maximum_nights_calendar_last_90d,

        
        SUM(CASE WHEN date >= CAST(:cutoff_date AS date) - INTERVAL '30 days'
                  AND available = 't' THEN 1 ELSE 0 END)  AS available_days_last_30d,
        ROUND(
            SUM(CASE WHEN date >= CAST(:cutoff_date AS date) - INTERVAL '30 days'
                      AND available = 't' THEN 1.0 ELSE 0 END)
            / NULLIF(
                SUM(CASE WHEN date >= CAST(:cutoff_date AS date) - INTERVAL '30 days'
                          THEN 1 ELSE 0 END),
                0
            ),
            4
        )                                                  AS available_rate_last_30d,
        AVG(CASE WHEN date >= CAST(:cutoff_date AS date) - INTERVAL '30 days'
                  THEN minimum_nights END)
                                                           AS avg_minimum_nights_calendar_last_30d,
        AVG(CASE WHEN date >= CAST(:cutoff_date AS date) - INTERVAL '30 days'
                  THEN maximum_nights END)
                                                           AS avg_maximum_nights_calendar_last_30d

    FROM core.calendar_day
    WHERE date >= CAST(:history_start_date AS date)
      AND date <= CAST(:cutoff_date AS date)
    GROUP BY listing_id;
    """,
    params={
        "history_start_date": history_start_date,
        "cutoff_date": cutoff_date,
    },
)

# convert numeric columns.
numeric_cal_cols = [
    "available_days_last_90d_total",
    "available_days_last_90d",
    "available_rate_last_90d",
    "avg_minimum_nights_calendar_last_90d",
    "avg_maximum_nights_calendar_last_90d",
    "available_days_last_30d",
    "available_rate_last_30d",
    "avg_minimum_nights_calendar_last_30d",
    "avg_maximum_nights_calendar_last_30d",
]
for col in numeric_cal_cols:
    calendar_features_all[col] = pd.to_numeric(calendar_features_all[col], errors="coerce")

assert calendar_features_all["listing_id"].duplicated().sum() == 0

print(calendar_features_all.shape)
print(calendar_features_all.head())

# write SQL to build one label row per listing.
# Use only dates after cutoff_date and up to label_end_date.

label_df = read_sql(
    """
    SELECT
        listing_id,
        COUNT(*)                                           AS future_calendar_days_observed_30d,
        SUM(CASE WHEN available = 't' THEN 1 ELSE 0 END)  AS future_available_days_30d,
        ROUND(
            SUM(CASE WHEN available = 't' THEN 1.0 ELSE 0 END) / NULLIF(COUNT(*), 0),
            4
        )                                                  AS future_available_rate_30d,
        CASE
            WHEN
                SUM(CASE WHEN available = 't' THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0)
                <= :threshold
            THEN 1
            ELSE 0
        END                                                AS high_demand_proxy
    FROM core.calendar_day
    WHERE date > CAST(:cutoff_date AS date)
      AND date <= CAST(:label_end_date AS date)
    GROUP BY listing_id;
    """,
    params={
        "cutoff_date": cutoff_date,
        "label_end_date": label_end_date,
        "threshold": HIGH_DEMAND_AVAILABLE_RATE_THRESHOLD,
    },
)

# convert numeric columns and make high_demand_proxy integer.
for col in ["future_calendar_days_observed_30d", "future_available_days_30d",
            "future_available_rate_30d"]:
    label_df[col] = pd.to_numeric(label_df[col], errors="coerce")

label_df["high_demand_proxy"] = label_df["high_demand_proxy"].astype(int)

assert label_df["listing_id"].duplicated().sum() == 0

print(label_df.shape)
print(label_df.head())

# Check label balance.

label_distribution = (
    label_df["high_demand_proxy"]
    .value_counts(dropna=False)
    .rename_axis("high_demand_proxy")
    .reset_index(name="count")
)

label_distribution["percentage"] = (
    label_distribution["count"] / label_distribution["count"].sum()
).round(4)

print(label_distribution)

# part 13..........................
# TODO: join static_features, review_features, calendar_features_all, and label_df.
feature_df = (
    label_df
    .merge(static_features, on="listing_id", how="inner")
    .merge(review_features, on="listing_id", how="left")
    .merge(calendar_features_all, on="listing_id", how="left")
)
# TODO: add cutoff_date and dataset_version columns.
feature_df["cutoff_date"] = str(cutoff_date)
feature_df["dataset_version"] = DATASET_VERSION

# TODO: fill missing review count features with zero.
review_count_cols = [
    "total_reviews_before_cutoff",
    "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff",
    "max_comment_len_before_cutoff",
]
for col in review_count_cols:
    if col in feature_df.columns:
        feature_df[col] = feature_df[col].fillna(0)

# TODO: handle missing days_since_last_review for listings with no reviews.
if "days_since_last_review" in feature_df.columns:
    feature_df["days_since_last_review"] = feature_df["days_since_last_review"].fillna(
        PAST_WINDOW_DAYS + 1
    )

assert feature_df["listing_id"].duplicated().sum() == 0

print(feature_df.shape)
print(feature_df.head())


protected_columns = {
    "listing_id",
    "cutoff_date",
    "dataset_version",
    "future_calendar_days_observed_30d",
    "future_available_days_30d",
    "future_available_rate_30d",
    "high_demand_proxy",
}

HIGH_MISSING_DROP_THRESHOLD = 0.95

# compute missing rates.
missing_rates = feature_df.isna().mean()

# find high-missing columns outside protected columns.
high_missing_cols = [
    col for col in feature_df.columns
    if col not in protected_columns
    and missing_rates[col] > HIGH_MISSING_DROP_THRESHOLD
]

# find constant columns outside protected columns.
constant_cols = [
    col for col in feature_df.columns
    if col not in protected_columns
    and feature_df[col].nunique(dropna=True) <= 1
]
# drop them from feature_df.
columns_to_drop = list(set(high_missing_cols + constant_cols))

print("Columns to drop:", columns_to_drop)

feature_df = feature_df.drop(columns=columns_to_drop)

print("New shape:", feature_df.shape)

duplicate_count = feature_df.duplicated(subset=["listing_id", "cutoff_date"]).sum()
missing_target_count = feature_df["high_demand_proxy"].isna().sum()
unique_target_values = sorted(feature_df["high_demand_proxy"].dropna().unique().tolist())

forbidden_columns = {
    "host_id",
    "host_pseudo_id",
    "reviewer_id",
    "reviewer_pseudo_id",
    "review_id",
    "license",
    "bathrooms_text",
}

present_forbidden_columns = sorted(forbidden_columns.intersection(feature_df.columns))

label_only_columns = [
    "future_calendar_days_observed_30d",
    "future_available_days_30d",
    "future_available_rate_30d",
    "high_demand_proxy",
]

model_input_columns = [
    col for col in feature_df.columns
    if col not in label_only_columns
    and col not in ["listing_id", "cutoff_date", "dataset_version"]
]

future_leakage_columns = [
    col for col in model_input_columns
    if col.startswith("future_")
]

# add asserts for each validation rule.
# for example:
# assert duplicate_count == 0
assert duplicate_count == 0, \
    f"Found {duplicate_count} duplicate (listing_id, cutoff_date) rows"

assert unique_target_values == [0, 1], \
    f"Target is not binary; found values: {unique_target_values}"

assert missing_target_count == 0, \
    f"Target has {missing_target_count} missing values"

assert len(present_forbidden_columns) == 0, \
    f"Forbidden PII columns present: {present_forbidden_columns}"

assert len(future_leakage_columns) == 0, \
    f"Future-leakage columns in model inputs: {future_leakage_columns}"

print("duplicate_count:", duplicate_count)
print("missing_target_count:", missing_target_count)
print("unique_target_values:", unique_target_values)
print("present_forbidden_columns:", present_forbidden_columns)
print("future_leakage_columns:", future_leakage_columns)
print("model_input_column_count:", len(model_input_columns))

missing_report = (
    feature_df
    .isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

missing_report.columns = ["column", "missing_rate"]

calendar_coverage_summary = (
    feature_df[["future_calendar_days_observed_30d"]]
    .describe()
    .reset_index()
)

print(missing_report.head(30))
print(label_distribution)
print(calendar_coverage_summary)


csv_path = FEATURE_DIR / f"listing_availability_features_{DATASET_VERSION}.csv"
parquet_path = FEATURE_DIR / f"listing_availability_features_{DATASET_VERSION}.parquet"
metadata_path = FEATURE_DIR / f"listing_availability_features_{DATASET_VERSION}_metadata.json"
validation_path = FEATURE_DIR / f"listing_availability_features_{DATASET_VERSION}_validation_report.json"
pii_audit_path = FEATURE_DIR / f"pii_audit_{DATASET_VERSION}.csv"

feature_df.to_csv(csv_path, index=False)
print("Saved CSV:", csv_path)

try:
    feature_df.to_parquet(parquet_path, index=False)
    print("Saved Parquet:", parquet_path)
except ImportError:
    print("Parquet not saved because pyarrow/fastparquet is not installed.")
    print("Install pyarrow with: pip install pyarrow")

# build metadata dictionary.
metadata = {
    "dataset_version": DATASET_VERSION,
    "entity_column": ENTITY_COLUMN,
    "cutoff_date": str(cutoff_date),
    "history_start_date": str(history_start_date),
    "label_end_date": str(label_end_date),
    "past_window_days": PAST_WINDOW_DAYS,
    "future_window_days": FUTURE_WINDOW_DAYS,
    "high_demand_available_rate_threshold": HIGH_DEMAND_AVAILABLE_RATE_THRESHOLD,
    "source_tables": [
        "core.listing",
        "core.host",
        "core.neighbourhood",
        "core.review",
        "core.calendar_day",
    ],
    "target_definition": (
        f"high_demand_proxy = 1 if future_available_rate_30d "
        f"<= {HIGH_DEMAND_AVAILABLE_RATE_THRESHOLD}; "
        "low availability proxy for high demand"
    ),
    "pii_exclusion_rules": (
        "Excluded from model inputs: host_id, host_pseudo_id, license, "
        "review_id, reviewer_id, reviewer_pseudo_id, bathrooms_text. "
        "listing_id kept as entity key only."
    ),
    "feature_columns": [
        col for col in feature_df.columns
        if col not in ["high_demand_proxy", "future_calendar_days_observed_30d",
                       "future_available_days_30d", "future_available_rate_30d"]
    ],
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# build validation_report dictionary.
validation_report = {
    "duplicate_listing_cutoff_rows": int(duplicate_count),
    "missing_target_count": int(missing_target_count),
    "target_values": unique_target_values,
    "present_forbidden_columns": present_forbidden_columns,
    "future_leakage_columns_in_model_inputs": future_leakage_columns,
    "missing_report": missing_report.to_dict(orient="records"),
    "label_distribution": label_distribution.to_dict(orient="records"),
    "calendar_coverage_summary": calendar_coverage_summary.to_dict(orient="records"),
}

with open(validation_path, "w", encoding="utf-8") as f:
    json.dump(validation_report, f, indent=2, ensure_ascii=False)

pii_audit.to_csv(pii_audit_path, index=False)

print("Saved metadata:", metadata_path)
print("Saved validation report:", validation_path)
print("Saved PII audit:", pii_audit_path)

print("Final shape:", feature_df.shape)

print(feature_df.head())

print(feature_df.info())

