# weather_utils.py

# pip install pandas pyarrow requests python-dateutil

# Helper methods to pull weather data from open-meteorology weather API

import requests
import pandas as pd
from pathlib import Path
from dateutil.parser import parse as parse_dt
from tqdm import tqdm

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

# City Name Normalization (Needed for Geocoding)
def to_proper_city_name(dataframe_key: str) -> str:
    overrides = {
        "los_angeles": "Los Angeles",
        "new_york_city": "New York City",
        "san_francisco": "San Francisco",
        "san_diego": "San Diego",
        "washington_DC": "Washington, United States",
        "washington_dc": "Washington, United States",
    }
    if dataframe_key in overrides:
        return overrides[dataframe_key]
    return dataframe_key.replace("_", " ").title()

# Geocode City Names
# Take a City name, and recieve lat/long/country/timezone info
# needed to retrieve weather information
def geocode_city(city_slug: str):
    query = to_proper_city_name(city_slug)
    request = requests.get(OPEN_METEO_GEOCODE, params={"name": query, "count": 1})
    request.raise_for_status()
    data = request.json()
    if not data.get("results"):
        raise ValueError(f"Could not geocode city '{query}' from slug '{city_slug}'")
    res = data["results"][0]
    return {
        "city": city_slug,
        "display_name": res.get("name", query),
        "lat": res["latitude"],
        "lon": res["longitude"],
        "country": res.get("country"),
        "timezone": res.get("timezone", "auto"),
    }

# Fetch Weather Information
def fetch_open_meteo_daily(lat, lon, start_date, end_date, timezone="auto"):
    # TThese variables are what the API will populate
    daily_vars = [
        "temperature_2m_max","temperature_2m_min","apparent_temperature_max","apparent_temperature_min",
        "precipitation_sum","rain_sum","snowfall_sum",
        "relative_humidity_2m_mean","cloud_cover_mean",
        "wind_speed_10m_max","wind_gusts_10m_max","shortwave_radiation_sum"
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone,
    }
    request = requests.get(OPEN_METEO_HIST, params=params)
    request.raise_for_status()
    payload = request.json()
    if "daily" not in payload:
        return pd.DataFrame(columns=["date"] + daily_vars)

    d = payload["daily"]
    df = pd.DataFrame(d)
    df = df.rename(columns={"time": "date"})  # Open-Meteo uses 'time' not dateTime, need to convert
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# convert the dates to actual datetime type
def infer_date_bounds(df: pd.DataFrame, date_col: str = "date") -> tuple[str, str]:
    if date_col not in df.columns:
        raise KeyError(f"Expected a '{date_col}' column in music dataframe.")
    dates = pd.to_datetime(df[date_col].map(lambda x: parse_dt(str(x)) if pd.notna(x) else pd.NaT), errors="coerce")
    start = dates.min()
    end = dates.max()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Could not infer min/max dates from the date column.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# For each city, get the weather data we retrieved and cached earlier
# build a joined dataframe of music -> weather data
# if weather data does not exist in the cache, get fresh data
def build_city_weather_table(song_df: pd.DataFrame,
                             city_list: list[str],
                             date_col: str = "date",
                             cache_dir: str = "data/weather_cache",
                             use_weather_cache: bool = True) -> pd.DataFrame:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    start_date, end_date = infer_date_bounds(song_df, date_col=date_col)
    print("--- BEGIN pulling weather data for cities---")

    frames = []
    for city in tqdm(city_list, desc="Fetching weather", unit=" city"):
        cache_path = Path(cache_dir) / f"{city}_{start_date}_{end_date}.parquet"
        if cache_path.exists() and use_weather_cache:
            city_weather = pd.read_parquet(cache_path)
        else:
            g = geocode_city(city)
            city_weather = fetch_open_meteo_daily(
                g["lat"], g["lon"], start_date, end_date, timezone=g["timezone"]
            )
            city_weather["city"] = city
            city_weather.to_parquet(cache_path, index=False)
        frames.append(city_weather)

    print("--- END pulling weather data for cities---")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# Build the music -> city -> weather dataset
# join music and weather together
def join_music_and_weather(unified_music_df: pd.DataFrame,
                           city_list: list[str],
                           date_col: str = "date",
                           use_weather_cache: bool = True) -> pd.DataFrame:
    df = unified_music_df.copy()

    if "city" not in df.columns:
        raise KeyError("Expected a 'city' column in music dataframe.")
    if date_col not in df.columns:
        raise KeyError(f"Expected a '{date_col}' column in music dataframe.")

    # normalize to Python date for exact match with weather
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    weather_df = build_city_weather_table(df, city_list, date_col=date_col, use_weather_cache=use_weather_cache)
    merged = df.merge(weather_df, left_on=["city", date_col], right_on=["city", "date"], how="left")

    # date_col is redundant if it still exists, so drop it
    if date_col != "date" and "date" in merged.columns:
        merged.drop(columns=["date"], inplace=True)

    return merged