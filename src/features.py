# Objective- turn clean df into values for the model
import pandas as pd

CATEGORICAL_COLS = ["Reporting_Airline", "Origin", "Dest"]
NUMERICAL_COLS = ["Month", "DayOfWeek", "DayofMonth", "DepHour"]
TARGET = "ArrDelay"


def build_encoders(df: pd.DataFrame) -> dict:
    # make map for encoders while reserving 0 for unknown/misc.
    encoders = {}
    for col in CATEGORICAL_COLS:
        categories = df[col].astype("category").cat.categories
        # start at 1; 0 = unknown/unseen bucket
        encoders[col] = {cat: idx + 1 for idx, cat in enumerate(categories)}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    # map encoders to indexes
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col + "_idx"] = df[col].map(encoders[col]).fillna(0).astype(int)
    return df


def compute_scaler(df: pd.DataFrame) -> dict:
    # acquire mean and standard dev for zscoring
    stats = {}
    for col in NUMERICAL_COLS:
        mean = float(df[col].mean())
        std = float(df[col].std()) or 1.0  # avoid divide-by-zero
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    # make data into z-scores for standardization
    df = df.copy()
    for col in NUMERICAL_COLS:
        df[col + "_scaled"] = (df[col] - scaler[col]["mean"]) / scaler[col]["std"]
    return df


def prepare_features(df: pd.DataFrame, encoders: dict = None, scaler: dict = None):
    # gives us full function
    if encoders is None:
        encoders = build_encoders(df)
    if scaler is None:
        scaler = compute_scaler(df)

    df = apply_encoders(df, encoders)
    df = apply_scaler(df, scaler)
    return df, encoders, scaler


def vocab_sizes(encoders: dict) -> dict:
    # Length of our encoders (how many)
    return {col: len(mapping) + 1 for col, mapping in encoders.items()}