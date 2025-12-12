# IMPORTANT
# ---------------------
# run pip install pyarrow before running file
import pandas as pd
from pathlib import Path
import music_dataset_utils as music_utils

# Pull data from all cities in list and save dataframe as parquet file
# The directory structure of the data is messy, this operation is expensive
# better to only perform it once
def build_music_dataset_and_save_to_parquet():
    print("---BEGIN Processing of Apple Music Dataset---")

    output_path = Path("data/script_outputs/music_dataset.parquet")

    city_list = ["atlanta","austin","chicago","dallas","denver","detroit","honolulu","houston","los_angeles","miami",
                 "new_york_city","philadelphia","san_francisco","san_diego","seattle","washington_DC"]

    all_dfs = []
    print("matching trending Apple Music data to Spotify Song Information...")
    for city in city_list:
        print("----- ", city, "data -----")
        city_df = music_utils.get_music_data_by_city(city, True)
        # dates are formatted as date : id , we only want the date portion
        city_df["date"] = (
            city_df["date"]
            .astype(str)
            .str.extract(r"^(\d{4}-\d{2}-\d{2})")[0]
        )
        city_df["date"] = pd.to_datetime(city_df["date"], errors="coerce").dt.date
        city_df["city"] = city
        all_dfs.append(city_df)

    unified_df = (pd.concat(all_dfs, ignore_index=True))

    # save as parquet file (not csv) for performance reasons
    unified_df.to_parquet(output_path, index=False, compression="zstd", engine="pyarrow")
    print(f"Saved unified dataset with {len(unified_df):,} rows to {output_path}")
    print("---END Processing of Apple Music Dataset---")
