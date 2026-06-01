import pandas as pd
from pathlib import Path
import typer
from airbnb_ops.config import PipelineConfig
from airbnb_ops.extract import read_csv_checked
from airbnb_ops.pii import handle_pii
from airbnb_ops.transform import build_neighbourhood_summary
from airbnb_ops.validate import validate_summary




def write_rep(rep_path:Path,out_path:Path,input_rows:int,
              input_colmns_l:pd.DataFrame,input_colmns_s:pd.DataFrame,
              output_colmn:pd.DataFrame,output_rows:int):
    rep=f"""#Airbnb Processing Report

    ## Overview

    Read inputs, handle PII, transform, validate and write result CSV file.

    ## Inputs Info

    - Number of input rows:{input_rows}
    - Listing input columns:{input_colmns_l}
    - Segment input columns:{input_colmns_s}

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
    - Number of output rows:{output_rows}
    - Output columns:{output_colmn}

    ## Output Files path
    - {rep_path}
    - {out_path}

    """
    rep_path.write_text(rep,encoding="utf-8")

app = typer.Typer()
@app.command()
def dummy():
    print("dummy")
    
@app.command()
def run():
    
    config = PipelineConfig()

    listings = read_csv_checked(
        config.listings_path
    )

    segments = read_csv_checked(
        config.segments_path
    )

    listings = handle_pii(listings)

    summary = build_neighbourhood_summary(
        listings,
        segments,
    )

    validate_summary(summary)
    config.output_path.parent.mkdir(
        parents=True,exist_ok=True
    )

    summary.to_csv(config.output_path,index=False)

    write_rep(config.report_path,config.output_path,len(listings),listings.columns,segments.columns,
              summary.columns,len(summary))
    
    typer.echo(
        f"Summary written to: {config.output_path}"
    )

    typer.echo(f"Report written to: {config.report_path}")

if __name__ == "__main__":
    app()