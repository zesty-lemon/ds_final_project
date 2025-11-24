import music_dataset_utils
import create_music_dataset
import build_music_weather_dataset
import music_weather_attribute_analysis

# Create Dataset files used for analysis
# if previously run on system, specify use_weather_cache = true for performance improvement
def create_dataset_files(use_music_cache: bool = False, use_weather_cache: bool = False):
    # Unzip Spotify Million Tracks Data if not already present (compressed to avoid github 100mb file limit)
    music_dataset_utils.unzip_spotify_million_tracks_csv_if_needed()

    # Pull the Music dataset from /data directory, unify it, and save to parquet files
    if not use_music_cache:
        create_music_dataset.build_music_dataset_and_save_to_parquet()

    # Pull Weather Data, Join to Music Data, and save as .parquet file
    build_music_weather_dataset.create_top_charts_song_data_weather_dataset(use_weather_cache=use_weather_cache)


# Perform Data Analysis on Music and Weather
# consisting of Plots and Statistical Tests
def perform_analysis_on_music_weather_data():
    # Create Plots relating to Weather's Influence on Music
    music_weather_attribute_analysis.run_music_weather_plots()
    # Perform Statistical Tests on Weather's Influence on Music
    music_weather_attribute_analysis.run_statistical_tests()

if __name__ == "__main__":
    # Prepare Data for Analysis
    # for performance, set both to true after running once
    create_dataset_files(use_music_cache = True, use_weather_cache = True)
    # Perform Data Analysis
    perform_analysis_on_music_weather_data()