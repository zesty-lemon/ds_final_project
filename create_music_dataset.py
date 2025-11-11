# IMPORTANT
# ---------------------
# run pip install pyarrow before running file
import pandas as pd
from pathlib import Path
import music_dataset_utils as music_utils

output_path = Path("data/script_outputs/music_dataset.parquet")


city_list = ["atlanta","austin","chicago","dallas","denver","detroit","honolulu","houston","los_angeles","miami",
             "new_york_city","philadelphia","san_francisco","san_diego","seattle","washington_DC"]

all_dfs = []

for city in city_list:
    print("----- ", city, "data -----")
    city_df = music_utils.get_music_data_by_city(city, True)
    all_dfs.append(city_df)

unified_df = (pd.concat(all_dfs, ignore_index=True))

# save as parquet file (not csv) for performance reasons
unified_df.to_parquet(output_path, index=False, compression="zstd", engine="pyarrow")

print(f"Saved unified dataset with {len(unified_df):,} rows to {output_path}")