import statistics
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype


def find_column_median(df: pd.DataFrame, column: str):
    """Only for numeric columns"""
    if not is_numeric_dtype(df[column]):
        raise TypeError("find_column_median function works only with numeric columns")

    data = df[column].dropna() # getting all non-null values from this column
    return statistics.median(data)

def find_column_mode(df: pd.DataFrame, column: str):
    """Only for string columns"""
    if not is_string_dtype(df[column]):
        raise TypeError("find_column_mode function works only with string columns")

    data = df[column].dropna()
    return statistics.mode(data)
