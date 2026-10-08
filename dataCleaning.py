import pandas as pd

df = pd.read_parquet("./data/raw/bts_2026.parquet")

keep = [
    # Date data known in advance
    "Month", "DayofMonth", "DayOfWeek", "CRSDepTime",
    # Airline/airport data known in advance
    "Reporting_Airline", "Origin", "Dest", "Distance",
    # Dependent variable
    "ArrDelay",
    # Used for cleaning
    "Cancelled", "Diverted",
]

# Trim Columns
df = df[keep].copy()

# Trim Rows
df = df[(df["Cancelled"]==0) & (df["Diverted"]==0)]

# Trim flights with unknown delays (No effect for bts_2026 parquet but good practice)
df = df.dropna(subset=["ArrDelay"])

# Convert formate 0000 time into just hours
df["DepHour"] = df["CRSDepTime"].astype(int)//100

# Drop cancelled, diverted, and CRSDepTime for final trimmed dataset
df = df.drop(columns=["Cancelled","Diverted","CRSDepTime"])


print(df.head())