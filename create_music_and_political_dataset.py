"""
Script for creating the primary dataset for political analysis
Grayson Causey
3 December, 2025
"""

from pathlib import Path
import pandas as pd
import time
import random
from pytrends.request import TrendReq

#
# Function for fetching GoogleTrends indices for political dataset
# Includes retry functionality as GoogleTrends is a little bit gimmicky,
# with Google locking you out if you attempt to retrieve data from the
# API to often or too much.
#
# Args:
#   - pytrends: the GoogleTrends client
#   - name: Batch label
#   - kw_list: the list of keywords to be used
#   - timeframe: range of dates to pull from
#
# Returns:
#   - None
#

def fetch_batch(pytrends, name, kw_list, timeframe):
    max_retries = 6
    for attempt in range(max_retries):
        try:
            print(f"\nFetching {name} (attempt {attempt + 1})")

            pytrends.build_payload(
                kw_list,
                timeframe=timeframe,
                geo="US"
            )

            df = pytrends.interest_over_time()

            if df.empty:
                raise ValueError("Too many attempts to retrieve data, wait a little bit.")

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df = df.add_prefix(name + "_")
            return df

        except Exception as e:
            print(f"Error fetching {name}: {e}")
            sleep_time = random.uniform(8, 16)
            print(f"Waiting {sleep_time:.1f} seconds before retrying...")
            time.sleep(sleep_time)

    print(f"FAILED to fetch {name} after {max_retries} attempts.\n")
    return None

#
# Function for merging GoogleTrends data with Music Dataset
#
# Args:
#   - None
#
# Returns:
#   - Finalized Pandas Dataframe as .parquet file
#

def create_music_and_politics_dataset():
    unified_path = Path("data/script_outputs/music_dataset.parquet")
    if not unified_path.exists():
        raise FileNotFoundError(
            f"Could not find {unified_path}. "
            "Run the music_dataset_utils script first to create the Parquet."
        )

    music_df = pd.read_parquet(unified_path)
    music_df["date"] = pd.to_datetime(music_df["date"], errors="coerce")

    music_columns = [
        "ID", "city", "date", "SONG", "ARTIST", "ALBUM",
        "danceability", "energy", "valence", "tempo"
    ]
    final_music = music_df[music_columns].copy()

    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = "2021-07-28 2022-01-31"

    # Batches to be used for political climate analysis.
    #
    # I felt that these topics were the most prevalent during the selected time period
    # What was most prevalent is somewhat subjective, and data can change if 
    # different key words are used
    
    batches = {
        "batch1_covid": [
            "covid", "covid vaccine", "mask mandate", "lockdown", "cdc"
        ],
        "batch2_biden_admin": [
            "biden", "joe biden", "white house",
            "build back better", "infrastructure bill"
        ],
        "batch3_economic": [
            "inflation", "gas prices", "economy",
            "supply chain", "shortage"
        ],
        "batch4_government_congress": [
            "congress", "senate", "house of representatives",
            "supreme court", "scotus"
        ],
        "batch5_social_issues": [
            "abortion law", "roe v wade",
            "texas abortion law", "immigration", "border crisis"
        ]
    }

    all_trends = []

    # Creating dataframe of batches
    for batch_name, kw_list in batches.items():
        df = fetch_batch(pytrends, batch_name, kw_list, timeframe)
        if df is not None:
            all_trends.append(df)

    # Formatting the batch dataframe
    google_trends_df = pd.concat(all_trends, axis=1)
    google_trends_df = google_trends_df.reset_index().rename(columns={"date": "date"})
    google_trends_df["date"] = pd.to_datetime(google_trends_df["date"])

    # Creating the individual batch columns
    batch_groups = {
        "covid_index":      [col for col in google_trends_df.columns if col.startswith("batch1_covid")],
        "biden_index":      [col for col in google_trends_df.columns if col.startswith("batch2_biden_admin")],
        "economic_index":   [col for col in google_trends_df.columns if col.startswith("batch3_economic")],
        "gov_index":        [col for col in google_trends_df.columns if col.startswith("batch4_government_congress")],
        "social_index":     [col for col in google_trends_df.columns if col.startswith("batch5_social_issues")]
    }

    combined_trends = pd.DataFrame()
    combined_trends["date"] = google_trends_df["date"]

    # Dividing batch values by 5, as each batch consists of 5 key words, this way we
    # retain an index of 0-100
    for name, cols in batch_groups.items():
        combined_trends[name] = google_trends_df[cols].sum(axis=1) / 5

    final_with_trends = final_music.merge(
        combined_trends,
        on="date",
        how="left"
    )

    # Saving newly created dataset as .parquet file
    out_path = Path("data/script_outputs/music_political.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_with_trends.to_parquet(
        out_path,
        index=False,
        compression="zstd",
        engine="pyarrow"
    )

if __name__ == "__main__":
    create_music_and_politics_dataset()