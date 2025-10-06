# Initial Proof of Concept
# Only look at NYC Data
# run pip install openpyxl first
import pandas as pd
import os

base_path = "data/"
music_dir = "/apple_music/20220119"

df = pd.DataFrame()
input_file_apple_music = os.path.join(base_path + music_dir, "new_york_city.xlsx")

df = pd.DataFrame()

df = pd.read_excel(input_file_apple_music)

print("done")



