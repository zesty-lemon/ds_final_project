import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import build_music_weather_dataset as dataset_builder

#
# Analysis of precipitation and music attributes (esp. danceability).
#
# In this script we:
#   1) Compute correlations between precipitation and music attributes.
#   2) Compute an overall correlation matrix for weather + music features.
#   3) Plot distributions of key variables.
#   4) Plot a LOWESS-smoothed relationship between danceability and precipitation.
#   5) Plot a 3-day rolling average of danceability by city.
#   6) Plot pairwise relationships between music attributes and weather.
#
def run_music_weather_plots():
    # ---------------------------------------------------------------------
    # Plot constants
    # ---------------------------------------------------------------------
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    # ---------------------------------------------------------------------
    # Data loading & basic prep
    # ---------------------------------------------------------------------
    df_music_weather = dataset_builder.read_music_weather_dataset()

    # Ensure dates are datetime
    df_music_weather["date"] = pd.to_datetime(df_music_weather["date"], errors="coerce")

    music_features = ["danceability", "energy", "valence", "tempo", "loudness"]
    weather_features = ["precipitation_sum", "temperature_2m_max", "cloud_cover_mean"]

    keep_cols = ["city", "date"] + weather_features + music_features
    df_music_weather = df_music_weather[keep_cols].copy()

    # Treat missing precipitation as 0
    df_music_weather["precipitation_sum"] = df_music_weather["precipitation_sum"].fillna(0)

    print(df_music_weather.head())

    # Aggregate to city-day level
    df_day = (
        df_music_weather
        .groupby(["city", "date"], as_index=False)
        .agg(
            {
                **{f: "mean" for f in music_features},
                **{w: "mean" for w in weather_features},
            }
        )
        .sort_values(["city", "date"])
    )

    def pretty_label(label: str) -> str:
        return label.replace("_", " ").title()

    # ---------------------------------------------------------------------
    # 1. Correlation Between Precipitation and Music Attributes
    # ---------------------------------------------------------------------
    corr_precip_music = df_day[["precipitation_sum"] + music_features].corr()
    corr_pretty = corr_precip_music.rename(index=pretty_label, columns=pretty_label)

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        corr_pretty,
        annot=True,
        cmap="coolwarm",
        center=0,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 10},
    )
    plt.title("Correlation Between Precipitation and Music Attributes", fontsize=16, pad=12)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 2. Overall Correlation Between Weather and Music Attributes
    # ---------------------------------------------------------------------
    corr = df_day[
        [
            "precipitation_sum",
            "temperature_2m_max",
            "cloud_cover_mean",
            "danceability",
            "energy",
            "valence",
            "tempo",
            "loudness",
        ]
    ].corr()

    renamed = {
        "precipitation_sum": "Precipitation",
        "temperature_2m_max": "Max Temp",
        "cloud_cover_mean": "Cloud Cover",
        "danceability": "Danceability",
        "energy": "Energy",
        "valence": "Valence",
        "tempo": "Tempo",
        "loudness": "Loudness",
    }

    corr_clean = corr.rename(index=renamed, columns=renamed)

    # plt.figure(figsize=(10, 8))
    sns.heatmap(corr_clean, annot=True, cmap="coolwarm", center=0)
    plt.title("Overall Correlation Between Weather and Music Attributes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 3. Distribution of Danceability by City
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_music_weather, x="city", y="danceability")
    ax.set_title("Distribution of Danceability by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Danceability Score")

    pretty_labels = [
        label.get_text().replace("_", " ").title()
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(pretty_labels, rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 4–6. Distributions of Daily Precipitation, Temperature, Cloud Cover
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df_day["precipitation_sum"], bins=50)
    plt.title("Distribution of Daily Precipitation Across all Cities")
    plt.xlabel("Daily Precipitation (mm)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df_day["temperature_2m_max"], bins=50)
    plt.title("Distribution of Daily Temperature Across All Cities")
    plt.xlabel("Daily Max Temperature")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df_day["cloud_cover_mean"], bins=50)
    plt.title("Distribution of Daily Cloud Cover Across All Cities")
    plt.xlabel("Daily Mean Cloud Cover")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 7. Danceability vs Precipitation (LOWESS Smoothed)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df_day,
        x="precipitation_sum",
        y="danceability",
        lowess=True,
        scatter_kws={"alpha": 0.3, "s": 20},
        line_kws={"lw": 3},
    )
    plt.xlabel("Daily Precipitation (mm)")
    plt.ylabel("Danceability")
    plt.title("Danceability vs Precipitation (LOWESS Smoothed)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 8. 3 - Day Rolling Average Danceability by City
    # ---------------------------------------------------------------------
    df_day["dance_roll3"] = (
        df_day
        .groupby("city")["danceability"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_day, x="date", y="dance_roll3", hue="city")
    plt.title("3 - Day Rolling Average Danceability by City")
    plt.xlabel("Date")
    plt.ylabel("Danceability (3-day rolling mean)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # 9. Relationships between music attributes and daily weather conditions
    # ---------------------------------------------------------------------
    # Note - I CANNOT get the title for this graph to show up
    # somewithing weird with Seabourne I do not understand
    # Title has to be added Manually
    cols = [
        "danceability",
        "valence",
        "energy",
        "precipitation_sum",
        "temperature_2m_max",
        "cloud_cover_mean",
    ]
    # Map these columns to pretty labels
    pretty_map = {c: pretty_label(c) for c in cols}
    # Same data but readable column names
    df_pretty = df_day.rename(columns=pretty_map)
    g = sns.pairplot(
        df_pretty,
        vars=list(pretty_map.values()),
        corner=True,
        plot_kws={"alpha": 0.3, "s": 10},
    )
    g.fig.suptitle("Relationships between music attributes and daily weather conditions", y=1.05)
    plt.show()



# Correlation on Delta (Change) in Weather vs Music
# rather than just values
def perform_delta_correlation():
    df_music_weather = dataset_builder.read_music_weather_dataset()
    df_music_weather["date"] = pd.to_datetime(df_music_weather["date"], errors="coerce")

    music_features = ["danceability", "energy", "valence"]
    weather_features = ["precipitation_sum", "temperature_2m_max", "cloud_cover_mean"]

    # Aggregate to city-day level
    df_day = (
        df_music_weather
        .groupby(["city", "date"], as_index=False)
        .agg(
            {
                **{f: "mean" for f in music_features},
                **{w: "mean" for w in weather_features},
            }
        )
        .sort_values(["city", "date"])
    )

    # compute day-to-day deltas within each city
    df_day[["delta_precip", "delta_temp", "delta_cloud"]] = (
        df_day.groupby("city")[["precipitation_sum", "temperature_2m_max", "cloud_cover_mean"]].diff()
    )

    df_day[["delta_dance", "delta_energy", "delta_valence"]] = (
        df_day.groupby("city")[["danceability", "energy", "valence"]].diff()
    )

    # remove NA deltas
    df_delta = df_day.dropna(subset=["delta_precip", "delta_dance"])

    # correlation of deltas
    print("Delta correlation:")
    print(df_delta[["delta_precip", "delta_dance", "delta_valence"]].corr())

def run_statistical_tests():
    # Correlation on Change in Precipitation vs Change in Dancability
    perform_delta_correlation()

if __name__ == "__main__":
    # Make plots for Music and Weather
    run_music_weather_plots()
    # Run Statistical Tests on Data
    run_statistical_tests()