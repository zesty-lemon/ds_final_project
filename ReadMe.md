# Influence of External Factors on Trending Music 
# An In-Depth Analysis

This project explores the relationship between weather conditions
and attributes of popular music tracks. Using historical weather
data and music features such as danceability, energy, and
tempo, the analysis aims to uncover meaningful correlations and
trends. 

Ultimately, the goal is to better understand how external
environmental factors may influence music characteristics or listener
preferences.

------------------------------------------------------------------------

## Project Goals

-   **Combine datasets** from Trending Top Charts (Apple Music),
    historical weather sources, and Spotify audio features.
-   **Perform exploratory data analysis (EDA)** on the merged dataset.
-   **Visualize relationships** between weather variables (e.g.,
    temperature, precipitation) and musical attributes (e.g., energy,
    danceability).
-   **Conduct statistical tests** to examine potential causal or
    correlational relationships between weather and music.

------------------------------------------------------------------------

## Run Instructions

**Pre-Recs**
### 1. Install Dependencies

``` bash
pip install -r requirements.txt
```
------------------------------------------------------------------------


**Instructions vary depending on the type of analysis you want to perform.
Refer to the appropriate section below.**


------------------------------------------------------------------------

## Weather vs. Music Analysis

### 1. Run Data Preparation & Main Analysis

Execute the orchestration script:

``` bash
python music_weather_analysis_orchestrator.py
```

By default, this script will:

-   Process raw data stored in `/data/` to produce a unified dataset.
-   Use cached music and weather data when available.
    -   To force regeneration, set `use_music_cache=False` and
        `use_weather_cache=False` inside the `create_dataset_files`
        function.
    -   This is not recommended due to slowness of the weather API and the historical nature of the data (nothing to be refreshed)
-   Perform EDA and statistical testing.
-   Generate plots and print key insights to
    the console.

------------------------------------------------------------------------

### 2. Optional: Run Attribute Analysis Directly

If you want to inspect or modify individual exploratory steps, you can
run the following script independently:

``` bash
python music_weather_attribute_analysis.py
```

------------------------------------------------------------------------

### 3. Output

-   Plots and statistical results will appear during execution.
-   Visual outputs can also be saved locally for later review within the
    `/plots/` directory.

------------------------------------------------------------------------
