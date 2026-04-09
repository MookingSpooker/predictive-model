import pandas as pd


def run_eda(df: pd.DataFrame, target: str = "Survived"):
    y = df[target]

    output: str = ""
    output += f"Shape: {df.shape}\n\n"

    output += f"Columns: {df.columns.tolist()}\n\n"

    output += f"Target distribution:\n{y.value_counts()}\n\n"

    output += f"Target ratio (%):\n{(y.value_counts(normalize=True) * 100).round(2)}\n\n"

    output += f"Dtypes: \n{df.dtypes}\n\n"

    output += f"Missing values:\n"
    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    output += f"{missing.sort_values('missing_count', ascending=False)}\n\n"

    output += f"Duplicates: {df.duplicated().sum()}\n\n"
    print(output)

def populate_nan_columns(df: pd.DataFrame, column: str, value: str | float | int):
    df[column] = df[column].fillna(value)

def remove_columns(df: pd.DataFrame, columns: list[str]):
    df.drop(columns=columns, inplace=True)

def remove_duplicates(df: pd.DataFrame):
    df.drop_duplicates(inplace=True)