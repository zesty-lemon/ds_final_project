# build_music_weather_dataset.py
# pip install pandas pyarrow requests python-dateutil statsmodels

from pathlib import Path
import pandas as pd
from weather_utils import join_music_and_weather


# create dataset of the apple music dataset joined with spotify data joined with weather data
def create_top_charts_song_data_weather_dataset(use_cache: bool = False):
    # Load unified music dataset
    unified_path = Path("data/script_outputs/music_dataset.parquet")
    if not unified_path.exists():
        raise FileNotFoundError(
            f"Could not find {unified_path}. "
            "Run the music_dataset_utils script first to create the Parquet."
        )

    unified_df = pd.read_parquet(unified_path)

    date_col = "date"

    # List of Cities on which to perform analysis
    city_list = ["atlanta", "austin", "chicago", "dallas", "denver", "detroit", "honolulu", "houston",
                 "los_angeles", "miami", "new_york_city", "philadelphia", "san_francisco",
                 "san_diego", "seattle", "washington_DC"]

    for city in city_list:
        g = unified_df[unified_df["city"] == city].copy()
        g["date"] = pd.to_datetime(g["date"])
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        missing = set(full) - set(g["date"])
        print(city, "missing days in MUSIC ONLY:", len(missing))


    # Join weather onto music rows
    merged_df = join_music_and_weather(unified_df, city_list=city_list, date_col=date_col, use_cache=use_cache)

    # Save merged dataset
    out_path = Path("data/script_outputs/music_with_weather.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(out_path, index=False, compression="zstd", engine="pyarrow")

    # Print a quick summary
    n_rows = len(merged_df)
    n_missing_weather = merged_df[
        "temperature_2m_max"].isna().sum() if "temperature_2m_max" in merged_df.columns else n_rows
    print(f"Saved merged dataset with weather: {n_rows:,} rows → {out_path}")
    print(f"Rows missing weather (date outside archive / geocode miss): {n_missing_weather:,}")


# read music dataset from file if it already exists
def read_music_weather_dataset(
    path: str | Path = "data/script_outputs/music_with_weather.parquet",
    engine: str = "pyarrow",
) -> pd.DataFrame:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Run `create_top_charts_song_data_weather_dataset()` first to generate it."
        )
    return pd.read_parquet(path, engine=engine)

if __name__ == "__main__":
    create_top_charts_song_data_weather_dataset()