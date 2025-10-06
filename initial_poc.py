# Initial Proof of Concept
# Only look at NYC Data
# run pip install openpyxl first
# a couple of methods were written by chatGPT, these are annotated by "# written by ChatGPT"
import re
import unicodedata

import pandas as pd
import os

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
    base_path = "data/"
    spotify_dir = "/spotify/"
    input_file_spotify = os.path.join(base_path + spotify_dir, "spotify_tracks_dataset.csv")
    df = pd.read_csv(input_file_spotify)
    print("Spotify Headers:")
    print(df.columns)
    return df


# normalize song names to join apple & spotify datasets
# written by ChatGPT
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


SINGLE_EP_SUFFIX = re.compile(r"\s*-\s*(single|ep)\b", flags=re.IGNORECASE)
def normalize_album_for_join(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKC", s)
    # drop bracketed qualifiers (deluxe), (remastered), [clean], etc.
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    # remove Apple-style suffixes like " - Single", " - EP"
    s = SINGLE_EP_SUFFIX.sub("", s)
    # keep letters/numbers from all scripts; drop punctuation
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Read Data into Dataframes
apple_music_df = get_apple_music_as_df()
spotify_df = get_spotify_tracks_as_df()

# Build normalized columns for Apple Music
apple_music_df = clean_apple_music_fields(apple_music_df)
apple_music_df["song_key"]   = apple_music_df["SONG"].map(normalize_for_join)
apple_music_df["album_key"] = apple_music_df["ALBUM"].map(normalize_album_for_join)

# apple_music_df["artist_key"] = apple_music_df["ARTIST"].map(normalize_for_join)

# Build normalized columns for Spotify
spotify_df["song_key"]   = spotify_df["track_name"].map(normalize_for_join)
spotify_df["album_key"] = spotify_df["album_name"].map(normalize_album_for_join)
# spotify_df["artist_key"] = spotify_df["artists"].map(normalize_for_join)

merged_df = apple_music_df.merge(
    spotify_df,
    how="left",
    on=["song_key", "album_key"],
    suffixes=("_apple", "_spotify")
)

# merged_df = apple_music_df.merge(
#     spotify_df,
#     on=["song_key", "artist_key"],
#     how="left",
#     suffixes=("_apple", "_spotify")
# )

# # Go date by date
# for date_value, group in apple_music_df.groupby('date_dt'):
#     print(f"\n=== {date_value} ===")
#     # print(group.head())


print("done")



