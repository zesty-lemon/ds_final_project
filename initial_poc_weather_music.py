# Initial Proof of Concept
# Only look at NYC Data
# run pip install openpyxl first
# a couple of methods were written by chatGPT, these are annotated by "# written by ChatGPT"
import re
import unicodedata

import pandas as pd
import os
import matplotlib.pyplot as plt

# read apple music dataset into dataframe
def get_apple_music_as_df() -> pd.DataFrame:
    base_path = "data/"
    music_dir = "/apple_music/20220119"
    input_file_apple_music = os.path.join(base_path + music_dir, "new_york_city.xlsx")
    df = pd.read_excel(input_file_apple_music)
    # file contains weird dates "07-21-2021 : 1".  Strip out the IDS ": 1" and convert to datetime
    print("Apple Music Headers:")
    print(df.columns)
    return df


# The apple music data is pretty dirty
# a lot of weird newline and space characters for some reason
# best practice is to fix it at import, but it is faster to strip it out later
# for the full project (not POC) we need to do better normilization of track, album, and artist names
def clean_apple_music_fields(df: pd.DataFrame) -> pd.DataFrame:
    def clean_text(x):
        if pd.isna(x):
            return ""
        s = str(x)
        # remove stray newline, tab, and backslash characters
        s = re.sub(r"[\n\r\t\\]+", " ", s)
        # normalize Unicode width and accent forms
        s = unicodedata.normalize("NFKC", s)
        # collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s
    # Clean Only these 3 columns
    for col in ["ARTIST", "ALBUM", "SONG"]:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
    # Apple dataset has weirdly formatted dates, we need to clean them up
    # date is date : trackId ( example: "07-01-2021 : 1")
    df['date_dt'] = pd.to_datetime(
        df['DATE'].astype(str).str.split(':').str[0].str.strip(), # get rid of everything after a :
        errors='coerce'
    )
    # music is only day-delineated, we do not need the time part of date-time
    df['date_dt'] = df['date_dt'].dt.date
    return df


# read spotify music dataset into dataframe
def get_spotify_tracks_as_df() -> pd.DataFrame:
    base_path = "data"
    spotify_dir = "/spotify/"
    input_file_spotify = os.path.join(base_path + spotify_dir, "spotify_million_tracks_data.csv")
    df = pd.read_csv(input_file_spotify)
    print("Spotify Headers:")
    print(df.columns)
    return df


# normalize song names to join apple & spotify datasets
def normalize_for_join(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    # normalize compatibility forms
    s = unicodedata.normalize("NFKC", s)
    # remove parentheses text like (feat. ...), [remix], etc.
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    # remove most punctuation but **keep letters/numbers from all languages**
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_and_clean_noaa_data() -> pd.DataFrame:
    file_path = "data/noaa/new_york_city/4138636.csv"
    df = pd.read_csv(file_path, low_memory=False)
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
    # average all daily temperatures across stations on that day
    daily = (
        df.groupby("DATE", as_index=False)
        .agg({"TAVG": "mean", "PRCP": "mean"})
        .rename(columns={"TAVG": "AVG_DAILY_TEMP", "PRCP": "AVG_DAILY_PRCP"})
        .sort_values("DATE")
    )
    return daily

# some simple NOAA plots
def plot_noaa_data(df: pd.DataFrame):
    # plot temperature
    plt.figure(figsize=(10, 5))
    plt.plot(df["DATE"], df["AVG_DAILY_TEMP"], label="Average Daily Temp")
    plt.title("Average Daily Temperature Over Time (New York City)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot precipitation
    plt.figure(figsize=(10, 5))
    plt.plot(df["DATE"], df["AVG_DAILY_PRCP"], label="Average Daily Precipitation")
    plt.title("Average Daily Precipitation Over Time (New York City)")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (in)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# get & return dancability by day
def get_danceability_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    danceability_col = next((c for c in merged_df.columns if c.lower().strip() == "danceability"), None)
    if danceability_col is None:
        raise KeyError("Couldn't find a 'danceability' column in Spotify data. "
                       "Check the CSV headers or rename the column to 'danceability'.")
    dedup = merged_df.drop_duplicates(subset=["date_dt", "song_key", "artist_key"])
    daily_danceability = (
        dedup
        .dropna(subset=["date_dt", danceability_col])
        .groupby("date_dt", as_index=False)[danceability_col]
        .mean()
        .rename(columns={danceability_col: "avg_danceability"})
        .sort_values("date_dt")
    )
    return daily_danceability

#written by chatGPT, I needed some quick & dirty plots to check if this was working
def quick_plots_and_corr(danceability_noaa: pd.DataFrame, rolling_window: int = 7) -> None:
    df = danceability_noaa.copy()

    # Correlations (Pearson & Spearman)
    num_cols = ["avg_danceability", "AVG_DAILY_TEMP", "AVG_DAILY_PRCP"]
    pearson_corr = df[num_cols].corr(method="pearson")
    spearman_corr = df[num_cols].corr(method="spearman")

    print("\n=== Correlations ===")
    print("Pearson:\n", pearson_corr.round(3).to_string())
    print("\nSpearman:\n", spearman_corr.round(3).to_string())

    # --- Simple lag diagnostics: does weather lead/lag danceability?
    # Negative lag => weather leads; Positive lag => danceability leads
    lags = [-7, -3, -1, 0, 1, 3, 7]
    lag_results = []
    for L in lags:
        shifted = df.copy()
        shifted["dance_shift"] = shifted["avg_danceability"].shift(-L)  # align so corr is between weather_t and dance_{t+L}
        lag_row = {
            "lag_days": L,
            "corr_temp": shifted["AVG_DAILY_TEMP"].corr(shifted["dance_shift"]),
            "corr_prcp": shifted["AVG_DAILY_PRCP"].corr(shifted["dance_shift"]),
        }
        lag_results.append(lag_row)

    print("\n=== Lag correlation (weather_t vs danceability_{t+lag}) ===")
    for r in lag_results:
        ct = "nan" if pd.isna(r["corr_temp"]) else f"{r['corr_temp']:.3f}"
        cp = "nan" if pd.isna(r["corr_prcp"]) else f"{r['corr_prcp']:.3f}"
        print(f"lag {r['lag_days']:>3}: temp={ct},  prcp={cp}")

    # rolling means to de-noise series
    df["dance_roll"] = df["avg_danceability"].rolling(rolling_window, min_periods=max(1, rolling_window // 2)).mean()
    df["temp_roll"]  = df["AVG_DAILY_TEMP"].rolling(rolling_window, min_periods=max(1, rolling_window // 2)).mean()
    df["prcp_roll"]  = df["AVG_DAILY_PRCP"].rolling(rolling_window, min_periods=max(1, rolling_window // 2)).mean()

    # === Plots ===
    # Danceability over time
    plt.figure(figsize=(10, 4))
    plt.plot(df["DATE"], df["avg_danceability"], label="Daily danceability")
    plt.plot(df["DATE"], df["dance_roll"], label=f"{rolling_window}-day rolling mean")
    plt.title("Average Danceability by Day")
    plt.xlabel("Date")
    plt.ylabel("Danceability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Temperature vs Danceability (scatter)
    plt.figure(figsize=(6, 5))
    plt.scatter(df["AVG_DAILY_TEMP"], df["avg_danceability"], s=12, alpha=0.7)
    plt.title("Danceability vs. Avg Daily Temperature")
    plt.xlabel("Avg Daily Temp (°C)")
    plt.ylabel("Danceability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Precipitation vs Danceability (scatter)
    plt.figure(figsize=(6, 5))
    plt.scatter(df["AVG_DAILY_PRCP"], df["avg_danceability"], s=12, alpha=0.7)
    plt.title("Danceability vs. Avg Daily Precipitation")
    plt.xlabel("Avg Daily Precipitation")
    plt.ylabel("Danceability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Rolling comparison plot (danceability vs temp rolling means)
    plt.figure(figsize=(10, 4))
    plt.plot(df["DATE"], df["dance_roll"], label="Danceability (roll)")
    plt.plot(df["DATE"], df["temp_roll"],  label="Temp (roll)")
    plt.title(f"Rolling Means ({rolling_window} days)")
    plt.xlabel("Date")
    plt.ylabel("Rolling values")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#---- Start Data Analysis ----

# --- Apple Music & Spotify Data ---
# Read Data into Dataframes
apple_music_df = get_apple_music_as_df()
spotify_df = get_spotify_tracks_as_df()

# Build normalized columns for Apple Music
apple_music_df = clean_apple_music_fields(apple_music_df)
apple_music_df["song_key"]   = apple_music_df["SONG"].map(normalize_for_join)
apple_music_df["artist_key"] = apple_music_df["ARTIST"].map(normalize_for_join)

# Build normalized columns for Spotify
spotify_df["song_key"]   = spotify_df["track_name"].map(normalize_for_join)
spotify_df["artist_key"] = spotify_df["artist_name"].map(normalize_for_join)

# search_df = spotify_df[spotify_df['track_name'].str.contains("Todo de ti", case=False, na=False)

merged_df = apple_music_df.merge(
    spotify_df,
    on=["song_key", "artist_key"],
    how="left",
    suffixes=("_apple", "_spotify")
)

merged_with_flag = apple_music_df.merge(
    spotify_df[["song_key", "artist_key", "track_name", "artist_name"]],
    on=["song_key", "artist_key"],
    how="left",
    suffixes=("_apple", "_spotify"),
    indicator=True
)

unmatched = merged_with_flag.loc[merged_with_flag["_merge"] == "left_only"].copy()
unique_unmatched_artists = unmatched["artist_key"].nunique(dropna=True)
print("Unmatched Artist: ",unique_unmatched_artists, " out of ", apple_music_df["artist_key"].nunique(dropna=True), " total")
# At this point any non-english songs do not match
# Apple Music has track titles/artist names in their original languages
# Spotify has them in english
# The set of mismatches isn't massive, so we will have to just manually go through them
# and make some mapping dictionary of japanese/korean -> english song titles/artist names
# for now, just drop rows that don't match (takes too long now, will fix later)
merged_df = merged_df.dropna(subset=['track_name'])

# --- NOAA Data ---
noaa_df = read_and_clean_noaa_data()
plot_noaa_data(noaa_df)


danceability_df = get_danceability_df(merged_df).rename(columns={"date_dt": "DATE"})

# make NOAA DATE match danceability_df's date
noaa_df["DATE"] = pd.to_datetime(noaa_df["DATE"], errors="coerce").dt.date

danceability_noaa = (
    noaa_df[["DATE", "AVG_DAILY_TEMP", "AVG_DAILY_PRCP"]]
    .merge(danceability_df, on="DATE", how="inner")
    .sort_values("DATE")
)

quick_plots_and_corr(danceability_noaa)

print("Done")



