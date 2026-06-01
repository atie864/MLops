import pandas as pd
import hashlib

DIRECT_PII_COLUMNS=["host_name"]

def pseudonymize_value(value, salt:str ='qbc12'):
    text=f"{salt}{value}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def handle_pii(df:pd.DataFrame):
    result=df.copy()
    # result=df
    result["host_key"]=(pseudonymize_value(str(result["host_id"])))
    result=result.drop(columns=["host_id"])
    result=result.drop(columns=DIRECT_PII_COLUMNS)
    return result
