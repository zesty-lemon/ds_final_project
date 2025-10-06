# Initial Proof of Concept
# Only look at NYC Data
# run pip install openpyxl first
import pandas as pd
import os

base_path = "data/"
music_dir = "/apple_music/20220119"

df = pd.DataFrame()
input_file_apple_music = os.path.join(base_path + music_dir, "new_york_city.xlsx")

df = pd.read_excel(input_file_apple_music)

# file contains weird dates "07-21-2021 : 1".  Strip out the IDS ": 1" and convert to datetime
df['date_dt'] = pd.to_datetime(
    df['DATE'].astype(str).str.split(':').str[0].str.strip(), # get rid of everything after a :
    errors='coerce'
)

print("done")



