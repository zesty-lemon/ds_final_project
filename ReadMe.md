# Influence of External Factors on Trending Music 
# An In-Depth Analysis

This project explores the relationship between weather conditions, 
political climate, demographic
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
------------------------------------------------------------------------

## Prerequisites

### Python Version
- Python 3.8 or higher

### Required Libraries

Install all dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
```

Or use the requirements file (if provided):

```bash
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

## Politics vs. Music Analysis

------------------------------------------------------------------------

### 1. Run Data Preparation

Execute the following script:

``` bash
python create_music_and_political_dataset.py
```

By default, this script will:
- Create a .parquet file that holds the merged political and musical data

------------------------------------------------------------------------

### 2. Run Main Analysis

Execute the analysis script:

``` bash
python music_politics_analysis.py
```

By default, this script will:
- Run the entire descriptive, explanatory, and descriptive analysis between politics and music
- Saves plots to /plots folder

------------------------------------------------------------------------

### 3. Output

- Plots will be displayed during execution
- They are also saved to /plots/ to be viewed when needed

------------------------------------------------------------------------
## Demographic vs. Music Analysis

------------------------------------------------------------------------
## Section Overview

This section analyzes the relationship between city demographics and music preferences using Apple Music chart data and U.S. Census Bureau demographic information. The analysis includes genre classification using machine learning models and correlation analysis between demographic factors and music tastes.

### API Keys & Credentials

1. **Google Cloud Translation API** (for Step 1)
   - Create a Google Cloud project
   - Enable Cloud Translation API
   - Download service account credentials JSON file
   - Set environment variable: `GOOGLE_APPLICATION_CREDENTIALS`

2. **U.S. Census Bureau API** (for Step 2)
   - Register for a free API key at: https://api.census.gov/data/key_signup.html
   - Update `CENSUS_API_KEY` in `Demographic_Data.py`

------------------------------------------------------------------------

## Execution Instructions

### Important: Run in Order

The scripts must be executed in the following sequence as each depends on outputs from the previous step.

------------------------------------------------------------------------
### Step 1: Translate Music Data

**Script:** `data_translator_final.py`

**Purpose:** Translates non-English song and artist names from Apple Music charts to English.

**Prerequisites:**
- Raw Apple Music data files in `selected_cities_apple/` folder
- Google Cloud Translation API credentials configured

**Run Command:**
Execute the following script:

``` bash
python data_tanslator.py
```
**What It Does:**
- Loads Apple Music chart data from multiple cities
- Detects non-English text in song names and artist names
- Translates text to English using Google Cloud Translation API
- Saves translated data to `selected_cities_apple/translated/` folder

**Expected Output:**
- Translated Excel files in `selected_cities_apple/translated/`
- Files named: `{city_name}_translated.xlsx`


------------------------------------------------------------------------

## Step 2: Collect Demographic Data

**Script:** `Demographic_Data.py`

**Purpose:** Retrieves demographic data for selected cities from U.S. Census Bureau.

**Prerequisites:**
- Census Bureau API key configured in the script
- List of cities defined in the script

**Run Command:**
Execute the Demographic Data script:

``` bash
python Demographic_Data.py
```

**What It Does:**
- Connects to U.S. Census Bureau API
- Retrieves demographic data (population, race, age, income, gender)
- Aggregates data from American Community Survey (ACS) 5-Year Estimates
- Calculates percentages for demographic categories
- 
**Expected Output:**
- `city_demographics.csv` in project root directory
- Contains columns: city, total_population, white, black, asian, hispanic, median_household_income, gender_male, gender_female, age groups, etc.


------------------------------------------------------------------------
### Step 3: Music Genre Classification Analysis

**Script:** `music_eda.py`

**Purpose:** Performs machine learning classification of music genres using audio features.

**Prerequisites:**
- Translated music data from Step 1 (`selected_cities_apple/translated/`)

**Run Command:**

Execute the analysis script:

``` bash
python music_eda.py
```
**What It Does:**
- Loads all translated music data
- Standardizes genre names across cities
- **Part 1:** Hip Hop vs R&B binary classification
  - Exploratory data analysis of audio features
  - Trains logistic regression models with 5-fold cross-validation
  - Generates ROC curves and performance metrics
- **Part 2:** Multi-genre pair classification
  - Evaluates all pairs of top 12 genres
  - Creates AUC matrix and hierarchical clustering
  - 
**Expected Output:**
- Folder: `outputs/genre_classification/`
- Visualizations:
  - `01_feature_distributions.png` - Audio feature histograms
  - `02_feature_correlation.png` - Feature correlation heatmap
  - `03_feature_importance_basic.png` - Feature discriminative power
  - `04_roc_curves_all_models.png` - ROC curves with cross-validation
  - `05_metrics_comparison.png` - Model performance comparison
  - `06_genre_pair_auc_heatmap.png` - Genre pair classification heatmap
- CSV files:
  - `model_performance_summary.csv`
  - `genre_pair_auc_matrix.csv`
  - `genre_pairs_ranked.csv`
------------------------------------------------------------------------
### Step 4: Combined Demographics & Music Analysis

*Script:** `music_demographic_eda.py`

**Purpose:** Analyzes correlations between city demographics and music preferences.

**Prerequisites:**
- Demographic data from Step 2 (`city_demographics.csv`)
- Translated music data from Step 1 (`selected_cities_apple/translated/`)

**Run Command:**
Execute the analysis script:
``` bash
python music_demographic_eda.py
```
**What It Does:**
- Loads demographic and music data
- **Demographic Visualizations:**
  - Population by city
  - Median household income
  - Racial/ethnic composition
  - Gender distribution
  - Age distribution
- **Music Visualizations:**
  - Songs per city
  - Genre distribution by city (stacked bar charts)
- **Correlation Analysis:**
  - Race demographics vs genre preferences
  - Age demographics vs genre preferences
- **Predictive Modeling:**
  - Predict genre percentages from demographics
  - Predict demographics from genre percentages
  - Multiple regression models (Ridge, Elastic Net, Random Forest, Gradient Boosting)
**Expected Output:**
- Folder: `outputs/eda_output/`
- Demographic visualizations:
  - `01_population_by_city.png`
  - `02_median_income_by_city.png`
  - `03_race_by_city.png`
  - `07_gender_by_city.png`
  - `12_age_brackets_by_city.png`
- Music visualizations:
  - `14_songs_per_city.png`
  - `18_genre_stacked_by_city.png`
  - `19_genre_percentage_by_city.png`
- Correlation analysis:
  - `20_race_genre_correlation.png`
  - `21_age_genre_correlation.png`
- Model performance:
  - `model_performance.csv`
  - `reverse_model_performance.csv`

------------------------------------------------------------------------

## Output Summary

### Generated Files
- `city_demographics.csv` - Demographic data for all cities
- `selected_cities_apple/translated/*.xlsx` - Translated music data

### Visualization Outputs
- **Genre Classification:** 7 images
- **Demographics & Music Analysis:** 10 images
- **CSV Summaries:** 5 CSV files with metrics and correlations

------------------------------------------------------------------------
