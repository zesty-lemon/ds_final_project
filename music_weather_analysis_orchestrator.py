import create_music_dataset
import build_music_weather_dataset
import music_weather_attribute_analysis

# Create Dataset files used for analysis
# if previously run on system, specify use_cache = true for performance improvement
def create_dataset_files(use_cache: bool = False):
    # Pull the Music dataset from /data directory, unify it, and save to parquet files
    create_music_dataset.build_music_dataset_and_save_to_parquet()
    # Pull Weather Data, Join to Music Data, and save as .parquet file
    build_music_weather_dataset.create_top_charts_song_data_weather_dataset(use_cache=use_cache)


# Perform Data Analysis on Music and Weather
# consisting of Plots and Statistical Tests
def perform_analysis_on_music_weather_data():
    # Create Plots relating to Weather's Influence on Music
    music_weather_attribute_analysis.run_music_weather_plots()
    # Perform Statistical Tests on Weather's Influence on Music
    music_weather_attribute_analysis.run_statistical_tests()

if __name__ == "__main__":
    # Prepare Data for Analysis
    create_dataset_files()
    # Perform Data Analysis
    perform_analysis_on_music_weather_data()