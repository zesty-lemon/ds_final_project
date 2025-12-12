"""
Visualization script for music and political climate analysis
Grayson Causey
December 4, 2025
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

def save_plot(name: str):
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}.png", dpi=300, bbox_inches="tight")

def pretty(label: str) -> str:
    return label.replace("_", " ").title()

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
def load_music_political_dataset():
    path = Path("data/script_outputs/music_political.parquet")
    if not path.exists():
        raise FileNotFoundError("music_political.parquet not found. Run dataset builder first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ---------------------------------------------------------------------
# 1. Political indices over time
# ---------------------------------------------------------------------
def plot_political_indices(df):

    daily = (
        df.groupby("date")[[
            "covid_index",
            "biden_index",
            "economic_index",
            "gov_index",
            "social_index"
        ]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    for col in daily.columns[1:]:
        sns.lineplot(
            data=daily,
            x="date",
            y=col,
            label=pretty(col),
            linewidth=2 if col == "covid_index" else 1.4,
            alpha=0.9
        )

    plt.title("Political Climate Indices Over Time", fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.tight_layout()
    save_plot("political_indices_timeseries")
    plt.show()

# ---------------------------------------------------------------------
# 2. Correlation heatmap: politics vs music
# ---------------------------------------------------------------------
def plot_political_music_correlations(df):

    daily = (
        df.groupby("date")
        .agg({
            "valence": "mean",
            "danceability": "mean",
            "energy": "mean",
            "tempo": "mean",
            "ID": "mean",
            "covid_index": "mean",
            "economic_index": "mean",
            "biden_index": "mean",
            "social_index": "mean",
            "gov_index": "mean",
        })
        .reset_index()
    )

    music_vars = ["valence", "danceability", "energy", "tempo", "ID"]
    pol_vars = ["covid_index", "economic_index", "biden_index", "social_index", "gov_index"]

    corr = daily[music_vars + pol_vars].corr().loc[pol_vars, music_vars]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Political Indices vs Music Attributes", fontweight="bold")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot("political_music_correlation")
    plt.show()

# ---------------------------------------------------------------------
# 3. Time series: music vs normalized political indices
# ---------------------------------------------------------------------
def plot_music_vs_politics_timeseries(df):

    daily = (
        df.groupby("date")
        .agg({
            "valence": "mean",
            "danceability": "mean",
            "energy": "mean",
            "covid_index": "mean",
            "economic_index": "mean",
            "biden_index": "mean",
            "social_index": "mean",
            "gov_index": "mean",
        })
        .reset_index()
    )

    pol_vars = ["covid_index", "economic_index", "biden_index", "social_index", "gov_index"]
    music_vars = ["valence", "danceability", "energy"]

    for col in pol_vars:
        daily[col] = (daily[col] - daily[col].min()) / (daily[col].max() - daily[col].min())

    plt.figure(figsize=(16, 14))

    for i, pol in enumerate(pol_vars, 1):
        plt.subplot(3, 2, i)

        for music in music_vars:
            sns.lineplot(data=daily, x="date", y=music, label=pretty(music))

        sns.lineplot(
            data=daily,
            x="date",
            y=pol,
            label=f"{pretty(pol)} (Normalized)",
            linewidth=2.5,
            alpha=0.7
        )

        plt.title(f"Music Attributes vs {pretty(pol)}", fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(fontsize=8)

    plt.tight_layout()
    save_plot("music_vs_political_timeseries")
    plt.show()

# ---------------------------------------------------------------------
# 4. Explanatory regressions (OLS)
# ---------------------------------------------------------------------
def plot_explanatory_regressions(df):

    daily = (
        df.groupby("date")
        .agg({
            "valence": "mean",
            "danceability": "mean",
            "energy": "mean",
            "covid_index": "mean",
            "economic_index": "mean",
            "biden_index": "mean",
            "social_index": "mean",
            "gov_index": "mean",
        })
        .reset_index()
    )

    music_targets = {
        "Danceability": "danceability",
        "Valence": "valence",
        "Energy": "energy"
    }

    pol_vars = ["covid_index", "economic_index", "biden_index", "social_index", "gov_index"]

    plt.figure(figsize=(24, 22))
    idx = 1

    for pol in pol_vars:
        for label, col in music_targets.items():

            plt.subplot(5, 3, idx)
            valid = daily[[pol, col]].dropna()

            model = LinearRegression().fit(valid[[pol]], valid[col])
            r2 = model.score(valid[[pol]], valid[col])

            sns.regplot(
                x=pol,
                y=col,
                data=daily,
                scatter_kws={"alpha": 0.4, "s": 25},
                line_kws={"color": "red"}
            )

            plt.title(f"{pretty(pol)} vs {label} (R²={r2:.2f})", fontsize=10)
            plt.xlabel(pretty(pol))
            plt.ylabel(label)

            idx += 1

    plt.tight_layout()
    save_plot("political_explanatory_regressions")
    plt.show()

# ---------------------------------------------------------------------
# 5. Predictive ROC curves (5-fold CV)
# ---------------------------------------------------------------------
def plot_predictive_roc(df):

    daily = (
        df.groupby("date")
        .agg({
            "valence": "mean",
            "covid_index": "mean",
            "economic_index": "mean",
            "biden_index": "mean",
            "social_index": "mean",
            "gov_index": "mean",
        })
        .reset_index()
        .dropna()
    )

    daily["high_valence"] = (daily["valence"] >= daily["valence"].median()).astype(int)

    X = daily[[
        "covid_index",
        "economic_index",
        "biden_index",
        "social_index",
        "gov_index"
    ]]
    y = daily["high_valence"]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plt.figure(figsize=(12, 10))

    for name, model in models.items():
        for i, (tr, val) in enumerate(cv.split(X, y), start=1):
            model.fit(X.iloc[tr], y.iloc[tr])
            probs = model.predict_proba(X.iloc[val])[:, 1]
            auc = roc_auc_score(y.iloc[val], probs)
            fpr, tpr, _ = roc_curve(y.iloc[val], probs)

            plt.plot(fpr, tpr, alpha=0.7, label=f"{name} Fold {i} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "5-Fold ROC Curves\nPredicting High/Low Valence from Political Indices",
        fontweight="bold"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot("predictive_roc_cv")
    plt.show()

def plot_random_forest_tuning(df):

    # Aggregate to daily level
    daily = (
        df.groupby("date")
        .agg({
            "valence": "mean",
            "covid_index": "mean",
            "economic_index": "mean",
            "biden_index": "mean",
            "social_index": "mean",
            "gov_index": "mean",
        })
        .reset_index()
        .dropna()
    )

    # Binary target
    daily["high_valence"] = (daily["valence"] >= daily["valence"].median()).astype(int)

    X = daily[
        ["covid_index", "economic_index", "biden_index", "social_index", "gov_index"]
    ]
    y = daily["high_valence"]

    # --------------------------------------------------
    # Random Forest + CV grid
    # --------------------------------------------------
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 300, 600],
        "max_depth": [3, 5, 8, None],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        scoring="roc_auc",
        cv=cv
    )

    grid.fit(X, y)

    # --------------------------------------------------
    # Prepare results for plotting
    # --------------------------------------------------
    results = pd.DataFrame(grid.cv_results_)

    plot_df = results[
        ["param_n_estimators", "param_max_depth", "mean_test_score"]
    ].rename(columns={
        "param_n_estimators": "Trees",
        "param_max_depth": "Max Depth",
        "mean_test_score": "Mean CV AUC"
    })

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(15, 11))

    sns.lineplot(
        data=plot_df,
        x="Trees",
        y="Mean CV AUC",
        hue="Max Depth",
        marker="o"
    )

    plt.title(
        "Random Forest Hyperparameter Tuning\n"
        "Predicting High/Low Valence from Political Indices",
        fontweight="bold"
    )

    plt.ylabel("Mean ROC AUC (Stratified 5-Fold CV)")
    plt.xlabel("Number of Trees")
    plt.legend(title="Max Depth")
    plt.tight_layout()

    save_plot("random_forest_hyperparameter_tuning")
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = load_music_political_dataset()

    plot_political_indices(df)
    plot_political_music_correlations(df)
    plot_music_vs_politics_timeseries(df)
    plot_explanatory_regressions(df)
    plot_predictive_roc(df)
    plot_random_forest_tuning(df)