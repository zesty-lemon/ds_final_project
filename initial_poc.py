# Initial Proof of Concept
# Only look at NYC Data
# run pip install openpyxl first
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
    df['date_dt'] = pd.to_datetime(
        df['DATE'].astype(str).str.split(':').str[0].str.strip(), # get rid of everything after a :
        errors='coerce'
    )
    # music is only day-delineated, we do not need the time part of date-time
    df['date_dt'] = df['date_dt'].dt.date
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

    for col in ["ARTIST", "ALBUM", "SONG"]:
        if col in df.columns:
            df[col] = df[col].map(clean_text)
    return df


# read spotify music dataset into dataframe
def get_spotify_tracks_as_df() -> pd.DataFrame:
    base_path = "data/"
    spotify_dir = "/spotify/"
    input_file_spotify = os.path.join(base_path + spotify_dir, "spotify_tracks_dataset.csv")
    df = pd.read_csv(input_file_spotify)
    return df


apple_music_df = get_apple_music_as_df()
spotify_df = get_spotify_tracks_as_df()


# Go date by date
for date_value, group in apple_music_df.groupby('date_dt'):
    print(f"\n=== {date_value} ===")
    # print(group.head())
print("Apple Music Headers:")
print(apple_music_df.columns)
print("Spotify Headers:")
print(spotify_df.columns)

# Clean & Sanitize datasets in preperation for joining
apple_music_df = clean_apple_music_fields(apple_music_df)

print("done")



