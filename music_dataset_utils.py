import os

import pandas as pd
import unicodedata
import re

def normalize_title(s):
    """Lowercase, strip whitespace, and remove punctuation."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    # Remove punctuation and special characters
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

apple_music_root = "data/apple_music/20220119/"
spotify_data_root = "data/apple_music/spotify_accoustic_features_of_tracks/"

city_list = ["atlanta"]

for city in city_list:
    apple_music_df = pd.read_excel(os.path.join(apple_music_root, city + ".xlsx"))
    spotify_df = pd.read_excel(os.path.join(spotify_data_root, city + ".xlsx"))
    spotify_df = spotify_df.rename(columns={spotify_df.columns[0]: "SONG"})

    apple_music_df["SONG"] = apple_music_df["SONG"].map(normalize_title)
    spotify_df["SONG"] = spotify_df["SONG"].map(normalize_title)

    merged_df = apple_music_df.merge(
        spotify_df,
        on="SONG",
        how="left",
        suffixes=("_apple", "_spotify")
    )

    unmatched_df = merged_df[merged_df["danceability"].isna()].copy()
    unmatched_unique = unmatched_df.drop_duplicates(subset=["SONG"]).copy()
    apple_unique = apple_music_df.drop_duplicates(subset=["SONG"]).copy()

    # Print summary
    total = len(apple_unique)
    unmatched = len(unmatched_unique)
    print(f"{city.title()}: {unmatched}/{total} unique unmatched songs ({unmatched/total:.1%})")

    # Save only unique unmatched songs
    output_path = os.path.join("data", f"{city}_unmatched_unique_titles.xlsx")
    unmatched_unique.to_excel(output_path, index=False)
    print(f"Saved unique unmatched songs to: {output_path}")