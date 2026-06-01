#Airbnb Processing Report

    ## Overview

    Read inputs, handle PII, transform, validate and write result CSV file.

    ## Inputs Info

    - Number of input rows:8
    - Listing input columns:Index(['listing_id', 'neighbourhood', 'price', 'minimum_nights',
       'availability_365', 'number_of_reviews', 'host_key'],
      dtype='object')
    - Segment input columns:Index(['neighbourhood', 'tourism_segment', 'priority_level'], dtype='object')

    ## PII Handling

    - Dropped: host_name
    - Replaced: host_id -> host_key

    ## Validation

    - output is not empty
    - required output columns exist
    - no PII columns exist
    - neighbourhood is not null
    - num_listings > 0
    - avg_price >= 0
    - availability_365_avg between 0 and 365

    - *Output validated successfully*

    ## Output info
    - Number of output rows:5
    - Output columns:Index(['neighbourhood', 'num_listings', 'avg_price', 'median_price',
       'avg_minimum_nights', 'availability_365_avg', 'total_reviews',
       'reviews_per_listing', 'tourism_segment', 'priority_level'],
      dtype='object')

    ## Output Files path
    - reports/hw01_a_run_report.md
    - data/processed/airbnb_neighbourhood_summary.csv

    