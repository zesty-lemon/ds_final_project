import os

import pandas as pd
import unicodedata
import re

# normalize helper.  Remove punctuation marks
def normalize_title(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # remove ft.
    s = s.replace("ft.", "")
    # Normalize accents/diacritics (e.g., "Beyoncé" -> "beyonce")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Remove punctuation and symbols
    s = re.sub(r"[^\w\s]", "", s)
    # Remove ALL spaces
    s = re.sub(r"\s+", "", s)
    # replace feat with with
    s = s.replace("feat", "with")

    return s.strip()


# read million tracks dataframe in from file
def get_spotify_million_tracks_df() -> pd.DataFrame:
    spotify_million_tracks_path = "data/spotify/spotify_million_tracks_data.csv"
    spotify_df = pd.read_csv(spotify_million_tracks_path, low_memory=False)

    # Normalize track name
    spotify_df["track_key"] = spotify_df["track_name"].map(normalize_title)

    # Keep only some columns
    cols_to_keep = ["track_key", "danceability", "energy", "loudness", "valence", "tempo"]
    spotify_df = spotify_df[cols_to_keep]

    return spotify_df


# filter down million tracks df to just the rows matching titles in unmatched_unique list of titles
def get_filtered_million_track_df(unmatched_unique: pd.DataFrame, million_tracks_df: pd.DataFrame) -> pd.DataFrame:
    unique_titles = unmatched_unique["SONG"].dropna().unique()
    matched_rows = []
    for song in unique_titles:
        # Find matching rows in the million tracks dataset
        matches = million_tracks_df[million_tracks_df["track_key"] == song]
        # If exactly one match add it to our list
        if len(matches) == 1:
            matched_rows.append(matches)

    if matched_rows:
        return pd.concat(matched_rows, ignore_index=True)
    # empty result with same columns if nothing matched
    return million_tracks_df.iloc[0:0].copy()


# take in a city name.  Then grab the appropriate apple music excel file and read it into a dataframe
# then join to the included spotify data.  However, this isn't complete.  Only ~90% of songs match this way
# to fill in all the remaining songs, do a second attempt at matching.  This time, try and match to the
# million tracks dataset.  Sometimes the match is messy (we only have title to go off of).  If > 1 matches we ignore
# if only one match we include.  This gets us around 96% of our songs having spotify data, which is enough
def get_music_and_data_df(city: str, apple_music_root: str, spotify_data_root: str, print_stats: bool) -> pd.DataFrame:
    apple_music_df = pd.read_excel(os.path.join(apple_music_root, city + ".xlsx"))
    spotify_df = pd.read_excel(os.path.join(spotify_data_root, city + ".xlsx"))
    spotify_df = spotify_df.rename(columns={spotify_df.columns[0]: "SONG"})
    million_tracks_df = get_spotify_million_tracks_df()
    apple_music_df["date"] = apple_music_df["DATE"]
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
    if print_stats:
        print(f"{city.title()} First Pass: {unmatched}/{total} unique unmatched songs ({unmatched/total:.1%})")

    second_pass_matches_df = get_filtered_million_track_df(unmatched_unique, million_tracks_df)
    if not second_pass_matches_df.empty:
        # merge on SONG (apple/spotify) == track_key (million tracks)
        merged_df = merged_df.merge(
            second_pass_matches_df,
            left_on="SONG",
            right_on="track_key",
            how="left",
            suffixes=("", "_mt")  # original keep bare; million-tracks get _mt
        )

        # For each feature column fill NAs from the million-tracks columns
        feature_cols = ["danceability", "energy", "loudness", "valence", "tempo"]
        for c in feature_cols:
            mt_col = f"{c}_mt"
            if mt_col in merged_df.columns:
                merged_df[c] = merged_df[c].fillna(merged_df[mt_col])

        # Clean up helper column(s)
        if "track_key" in merged_df.columns:
            merged_df.drop(columns=["track_key"], inplace=True)
        for c in feature_cols:
            mt_col = f"{c}_mt"
            if mt_col in merged_df.columns:
                merged_df.drop(columns=[mt_col], inplace=True)

    if print_stats:
        unmatched_df = merged_df[merged_df["danceability"].isna()].copy()
        unmatched_unique = unmatched_df.drop_duplicates(subset=["SONG"]).copy()
        unmatched = len(unmatched_unique)
        print(f"{city.title()} Second Pass: {unmatched}/{total} unique unmatched songs ({unmatched/total:.1%})")
    return merged_df


# public method.  get matched df based on city name
# WARNING: may contain N/A values for music attributes for songs that don't match spotify data
# be sure to handle when using
def get_music_data_by_city(city_name: str, print_stats: bool = True) -> pd.DataFrame:
    apple_music_root = "data/apple_music/20220119/"
    spotify_data_root = "data/apple_music/spotify_accoustic_features_of_tracks/"

    city_df = get_music_and_data_df(city_name, apple_music_root, spotify_data_root, print_stats)
    return city_df
