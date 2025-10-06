import pandas as pd

file_path = "data/noaa/new_york_city/4138636.csv"
df = pd.read_csv(file_path, low_memory=False)
# keep only relevant columns ---
df = df[["DATE", "STATION", "TMAX", "TMIN", "TAVG", "PRCP"]]
# force datetime
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
# force numeric on temp columns
for col in ["TMAX", "TMIN", "TAVG", "PRCP"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# manually fudge missing temp values
df["TAVG"] = df["TAVG"].fillna((df["TMAX"] + df["TMIN"]) / 2)
# drop rows with no valid TAVG
df = df.dropna(subset=["TAVG", "DATE"])
#average all daily temperatures across stations on that day
daily = (
    df.groupby("DATE", as_index=False)
      .agg({"TAVG": "mean", "PRCP": "mean"})
      .rename(columns={"TAVG": "AVG_DAILY_TEMP", "PRCP": "AVG_DAILY_PRCP"})
      .sort_values("DATE")
)

print(daily.head())
print(f"\nNumber of days with temperature data: {len(daily)}")
