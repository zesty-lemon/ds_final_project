import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# File paths
DEMOGRAPHICS_FILE = '/Users/albinmeli/CS5870/ds_final_project/city_demographics.csv'
TRANSLATED_FOLDER = '/Users/albinmeli/CS5870/ds_final_project/selected_cities_apple/translated'
OUTPUT_FOLDER = '/Users/albinmeli/CS5870/ds_final_project/eda_output'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_demographics():
    """Load city demographics data."""
    print("Loading city demographics...")
    df = pd.read_csv(DEMOGRAPHICS_FILE)
    print(f"Loaded {len(df)} cities")
    return df

def standardize_genres(df):
    """Standardize genre names for consistency."""
    if 'genre' in df.columns:
        df['genre'] = (df['genre']
                      .str.lower()
                      .str.strip()
                      .str.replace('hip-hop', 'hip hop', regex=False)
                      .str.replace('r&b|r & b', 'r and b', regex=True))
    return df

def load_music_data():
    """Load and standardize music data from all cities."""
    print("\nLoading music data...")
    music_data = {}
    translated_path = Path(TRANSLATED_FOLDER)
    
    if not translated_path.exists():
        print(f"Folder not found: {translated_path}")
        return music_data
    
    # Load CSV and XLSX files
    for file in sorted(translated_path.glob('*.csv')):
        city_name = file.stem.replace('_translated', '')
        try:
            df = pd.read_csv(file)
            df = standardize_genres(df)
            music_data[city_name] = df
            print(f"{city_name}: {len(df)} songs")
        except Exception as e:
            print(f"{city_name}: Failed - {e}")
    
    for file in sorted(translated_path.glob('*.xlsx')):
        city_name = file.stem.replace('_translated', '')
        if city_name not in music_data:  # Only load if not already loaded as CSV
            try:
                df = pd.read_excel(file)
                df = standardize_genres(df)
                music_data[city_name] = df
                print(f"{city_name}: {len(df)} songs")
            except Exception as e:
                print(f"{city_name}: Failed - {e}")
    
    print(f"\nLoaded {len(music_data)} cities")
    return music_data

# ============================================================================
# DEMOGRAPHIC VISUALIZATIONS
# ============================================================================

def plot_population(df):
    """Bar chart of population by city."""
    if 'total_population' not in df.columns or 'city' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(df['city'], df['total_population'], color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('City')
    ax.set_ylabel('Total Population')
    ax.set_title('Population by City', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/01_population_by_city.png', dpi=300)
    print("Saved: 01_population_by_city.png")
    plt.close()

def plot_income(df):
    """Bar chart of median household income by city."""
    income_cols = [col for col in df.columns if 'median' in col.lower() and 'income' in col.lower()]
    
    if not income_cols or 'city' not in df.columns:
        return
    
    income_col = income_cols[0]
    fig, ax = plt.subplots(figsize=(14, 6))
    
    incomes = df[income_col]
    bars = ax.bar(df['city'], incomes, 
                  color=plt.cm.RdYlGn(incomes / incomes.max()), 
                  alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('City')
    ax.set_ylabel('Median Household Income ($)')
    ax.set_title('Median Household Income by City', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add mean line
    mean_income = incomes.mean()
    ax.axhline(mean_income, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: ${mean_income:,.0f}')
    ax.legend()
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/02_median_income_by_city.png', dpi=300)
    print("Saved: 02_median_income_by_city.png")
    plt.close()

def plot_race_demographics(df):
    """Line plot showing racial composition across cities."""
    race_cols = ['white', 'black', 'asian', 'hispanic']
    race_labels = ['White', 'Black', 'Asian', 'Hispanic']
    
    if not all(col in df.columns for col in race_cols):
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    df_sorted = df.sort_values('total_population' if 'total_population' in df.columns else 'city')
    x = np.arange(len(df_sorted))
    
    # Plot with different markers and line styles
    styles = [('o', '-'), ('s', '--'), ('^', '-.'), ('D', ':')]
    colors = ['#2E8B57', '#DC143C', '#4169E1', '#FF8C00']
    
    for race_col, race_label, (marker, linestyle), color in zip(race_cols, race_labels, styles, colors):
        ax.plot(x, df_sorted[race_col], marker=marker, linestyle=linestyle,
               linewidth=2.5, markersize=9, color=color, label=race_label, 
               alpha=0.85, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['city'], rotation=45, ha='right')
    ax.set_xlabel('City', fontweight='bold')
    ax.set_ylabel('Population Count', fontweight='bold')
    ax.set_title('Racial/Ethnic Population Across Cities', fontweight='bold')
    ax.legend(title='Race/Ethnicity', framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/03_race_by_city.png', dpi=300)
    print("Saved: 03_race_by_city.png")
    plt.close()

def plot_gender_demographics(df):
    """Grouped bar chart of gender distribution."""
    if not all(col in df.columns for col in ['gender_male', 'gender_female', 'city']):
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['gender_male'], width, label='Male', 
          alpha=0.8, color='steelblue', edgecolor='black')
    ax.bar(x + width/2, df['gender_female'], width, label='Female', 
          alpha=0.8, color='coral', edgecolor='black')
    
    ax.set_xlabel('City')
    ax.set_ylabel('Population')
    ax.set_title('Gender Distribution by City', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['city'], rotation=45, ha='right')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/07_gender_by_city.png', dpi=300)
    print("Saved: 07_gender_by_city.png")
    plt.close()

def plot_age_demographics(df):
    """Stacked bar chart of age distribution."""
    age_cols = ['age_under_18_pct', 'age_18_to_64_pct', 'age_65_plus_pct']
    
    if not all(col in df.columns for col in age_cols) or 'city' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df))
    
    colors = ['#FFA07A', '#90EE90', '#87CEEB']
    labels = ['Under 18', '18-64', '65+']
    
    under_18 = df['age_under_18_pct'].values
    age_18_64 = df['age_18_to_64_pct'].values
    age_65_plus = df['age_65_plus_pct'].values
    
    ax.bar(x, under_18, label=labels[0], color=colors[0], alpha=0.8, edgecolor='black')
    ax.bar(x, age_18_64, bottom=under_18, label=labels[1], color=colors[1], alpha=0.8, edgecolor='black')
    ax.bar(x, age_65_plus, bottom=under_18 + age_18_64, label=labels[2], color=colors[2], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('City')
    ax.set_ylabel('Percentage')
    ax.set_title('Age Distribution by City', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['city'], rotation=45, ha='right')
    ax.legend(title='Age Group')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/12_age_brackets_by_city.png', dpi=300)
    print("Saved: 12_age_brackets_by_city.png")
    plt.close()

# ============================================================================
# MUSIC VISUALIZATIONS
# ============================================================================

def plot_songs_per_city(music_data):
    """Bar chart of song counts by city."""
    if not music_data:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    cities = list(music_data.keys())
    counts = [len(df) for df in music_data.values()]
    
    bars = ax.bar(cities, counts, color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Songs')
    ax.set_title('Songs in Chart by City', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/14_songs_per_city.png', dpi=300)
    print("Saved: 14_songs_per_city.png")
    plt.close()

def plot_genre_by_city(music_data):
    """Stacked bar charts showing genre distribution."""
    if not music_data:
        return
    
    # Get top 10 genres across all cities
    all_genres = []
    for df in music_data.values():
        if 'genre' in df.columns:
            all_genres.extend(df['genre'].dropna().tolist())
    
    if not all_genres:
        return
    
    top_genres = pd.Series(all_genres).value_counts().head(10).index.tolist()
    
    # Build genre matrix
    genre_matrix = []
    cities = []
    for city, df in music_data.items():
        if 'genre' in df.columns:
            cities.append(city)
            genre_counts = df['genre'].value_counts()
            row = [genre_counts.get(genre, 0) for genre in top_genres]
            genre_matrix.append(row)
    
    genre_df = pd.DataFrame(genre_matrix, columns=top_genres, index=cities)
    
    # Absolute counts
    fig, ax = plt.subplots(figsize=(16, 8))
    genre_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, edgecolor='black')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Songs')
    ax.set_title('Genre Composition by City (Top 10 Genres)', fontweight='bold')
    ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/18_genre_stacked_by_city.png', dpi=300)
    print("Saved: 18_genre_stacked_by_city.png")
    plt.close()
    
    # Percentage
    genre_pct = genre_df.div(genre_df.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(16, 8))
    genre_pct.plot(kind='bar', stacked=True, ax=ax, width=0.8, edgecolor='black')
    ax.set_xlabel('City')
    ax.set_ylabel('Percentage')
    ax.set_title('Genre Distribution by City (Normalized %)', fontweight='bold')
    ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/19_genre_percentage_by_city.png', dpi=300)
    print("Saved: 19_genre_percentage_by_city.png")
    plt.close()

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def normalize_city_name(city_name):
    """Normalize city names for matching between datasets."""
    normalized = city_name.replace('_', ' ').title()
    special_cases = {
        'New York City': 'New York',
        'Washington Dc': 'Washington',
    }
    return special_cases.get(normalized, normalized)

def create_combined_dataset(demographics_df, music_data, top_genres):
    """Combine demographics and genre percentages into one dataset."""
    combined_data = []
    
    for city_key, music_df in music_data.items():
        normalized_city = normalize_city_name(city_key)
        city_demo = demographics_df[demographics_df['city'].str.lower() == normalized_city.lower()]
        
        if len(city_demo) == 0 or 'genre' not in music_df.columns:
            continue
        
        city_row = {'city': normalized_city}
        
        # Add demographic features
        demo_cols = ['white_pct', 'black_pct', 'asian_pct', 'hispanic_pct',
                     'age_under_18_pct', 'age_18_to_64_pct', 'age_65_plus_pct']
        for col in demo_cols:
            if col in city_demo.columns:
                city_row[col] = city_demo.iloc[0][col]
        
        # Add genre percentages
        total_songs = len(music_df)
        genre_counts = music_df['genre'].value_counts()
        for genre in top_genres:
            count = genre_counts.get(genre, 0)
            safe_name = genre.replace(' ', '_').replace('&', 'and').replace('-', '_')
            city_row[f'genre_{safe_name}_pct'] = (count / total_songs * 100) if total_songs > 0 else 0
        
        combined_data.append(city_row)
    
    return pd.DataFrame(combined_data)

def plot_correlation_heatmap(combined_df, x_cols, y_cols, title, filename):
    """Create correlation heatmap between two sets of variables."""
    if len(combined_df) < 3:
        print(f"Not enough data for {filename}")
        return
    
    corr_matrix = pd.DataFrame(index=y_cols, columns=x_cols)
    for y_col in y_cols:
        for x_col in x_cols:
            corr_matrix.loc[y_col, x_col] = combined_df[x_col].corr(combined_df[y_col])
    
    corr_matrix = corr_matrix.astype(float)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'},
               linewidths=1, linecolor='white', ax=ax)
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_FOLDER}/{filename}', dpi=300)
    print(f"Saved: {filename}")
    plt.close()

def analyze_correlations(demographics_df, music_data):
    """Analyze correlations between demographics and music preferences."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    if not music_data:
        return
    
    # Get top 8 genres
    all_genres = []
    for df in music_data.values():
        if 'genre' in df.columns:
            all_genres.extend(df['genre'].dropna().tolist())
    
    top_genres = pd.Series(all_genres).value_counts().head(8).index.tolist()
    combined_df = create_combined_dataset(demographics_df, music_data, top_genres)
    
    if len(combined_df) < 3:
        print("Insufficient data for correlation analysis")
        return
    
    # Race vs Genre
    race_cols = ['white_pct', 'black_pct', 'asian_pct', 'hispanic_pct']
    genre_cols = [col for col in combined_df.columns if col.startswith('genre_') and col.endswith('_pct')]
    
    available_race = [col for col in race_cols if col in combined_df.columns]
    if available_race and genre_cols:
        plot_correlation_heatmap(
            combined_df, available_race, genre_cols,
            'Correlation: Race Demographics vs Music Genre Preferences',
            '20_race_genre_correlation.png'
        )
    
    # Age vs Genre
    age_cols = ['age_under_18_pct', 'age_18_to_64_pct', 'age_65_plus_pct']
    available_age = [col for col in age_cols if col in combined_df.columns]
    if available_age and genre_cols:
        plot_correlation_heatmap(
            combined_df, available_age, genre_cols,
            'Correlation: Age Demographics vs Music Genre Preferences',
            '21_age_genre_correlation.png'
        )

# ============================================================================
# PREDICTIVE MODELING
# ============================================================================

def train_and_evaluate_models(X_train, X_test, Y_train, Y_test, target_cols):
    """Train multiple regression models and return results."""
    models = {
        'Elastic Net': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=2000),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }
    
    all_results = {}
    
    for model_name, base_model in models.items():
        print(f"\nTraining: {model_name}")
        target_results = {}
        
        for target_col in target_cols:
            from sklearn.base import clone
            model = clone(base_model)
            model.fit(X_train, Y_train[target_col])
            
            y_pred = model.predict(X_test)
            y_test = Y_test[target_col]
            
            target_results[target_col] = {
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        all_results[model_name] = target_results
        avg_r2 = np.mean([r['r2'] for r in target_results.values()])
        print(f"  Average R²: {avg_r2:.3f}")
    
    return all_results

def predict_genre_from_demographics(demographics_df, music_data):
    """Build models to predict genre percentages from demographics."""
    print("\n" + "="*80)
    print("PREDICTING GENRE FROM DEMOGRAPHICS")
    print("="*80)
    
    if not music_data:
        return
    
    # Get top 5 genres
    all_genres = []
    for df in music_data.values():
        if 'genre' in df.columns:
            all_genres.extend(df['genre'].dropna().tolist())
    
    top_genres = pd.Series(all_genres).value_counts().head(5).index.tolist()
    model_df = create_combined_dataset(demographics_df, music_data, top_genres)
    
    if len(model_df) < 6:
        print("Insufficient data for modeling")
        return
    
    # Prepare data
    feature_cols = [col for col in model_df.columns 
                   if col not in ['city'] and not col.startswith('genre_')]
    target_cols = [col for col in model_df.columns 
                  if col.startswith('genre_') and col.endswith('_pct')]
    
    X = model_df[feature_cols].fillna(model_df[feature_cols].mean())
    Y = model_df[target_cols]
    
    # Split and scale
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    all_results = train_and_evaluate_models(X_train_scaled, X_test_scaled, 
                                           Y_train, Y_test, target_cols)
    
    # Save comparison
    comparison_data = []
    for model_name, results in all_results.items():
        r2_scores = [max(0, r['r2']) for r in results.values()]
        comparison_data.append({
            'Model': model_name,
            'Average R²': np.mean(r2_scores),
            'Average RMSE (%)': np.mean([r['rmse'] for r in results.values()])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{OUTPUT_FOLDER}/model_performance.csv', index=False)
    print("\nSaved: model_performance.csv")
    print(comparison_df.to_string(index=False))

def predict_demographics_from_genre(demographics_df, music_data):
    """Build models to predict demographics from genre percentages."""
    print("\n" + "="*80)
    print("PREDICTING DEMOGRAPHICS FROM GENRE")
    print("="*80)
    
    if not music_data:
        return
    
    # Use moderate number of genres to avoid overfitting
    all_genres = []
    for df in music_data.values():
        if 'genre' in df.columns:
            all_genres.extend(df['genre'].dropna().tolist())
    
    num_cities = len(music_data)
    max_genres = max(5, num_cities // 2)
    top_genres = pd.Series(all_genres).value_counts().head(max_genres).index.tolist()
    
    model_df = create_combined_dataset(demographics_df, music_data, top_genres)
    
    if len(model_df) < 10:
        print("Insufficient data for modeling")
        return
    
    # Prepare data (genres as features, demographics as targets)
    feature_cols = [col for col in model_df.columns 
                   if col.startswith('genre_') and col.endswith('_pct')]
    target_cols = ['white_pct', 'black_pct', 'asian_pct', 'hispanic_pct',
                   'age_under_18_pct', 'age_18_to_64_pct', 'age_65_plus_pct']
    target_cols = [col for col in target_cols if col in model_df.columns]
    
    X = model_df[feature_cols].fillna(0)
    Y = model_df[target_cols]
    
    # Split and scale
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    all_results = train_and_evaluate_models(X_train_scaled, X_test_scaled, 
                                           Y_train, Y_test, target_cols)
    
    # Save comparison
    comparison_data = []
    for model_name, results in all_results.items():
        r2_scores = [max(0, r['r2']) for r in results.values()]
        comparison_data.append({
            'Model': model_name,
            'Average R²': np.mean(r2_scores),
            'Average RMSE (%)': np.mean([r['rmse'] for r in results.values()])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{OUTPUT_FOLDER}/reverse_model_performance.csv', index=False)
    print("\nSaved: reverse_model_performance.csv")
    print(comparison_df.to_string(index=False))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete analysis pipeline."""
    print("\n" + "="*80)
    print("MUSIC & DEMOGRAPHIC EDA")
    print("="*80)
    
    # Load data
    demographics_df = load_demographics()
    music_data = load_music_data()
    
    # Demographic visualizations
    print("\nGenerating demographic visualizations...")
    plot_population(demographics_df)
    plot_income(demographics_df)
    plot_race_demographics(demographics_df)
    plot_gender_demographics(demographics_df)
    plot_age_demographics(demographics_df)
    
    # Music visualizations
    if music_data:
        print("\nGenerating music visualizations...")
        plot_songs_per_city(music_data)
        plot_genre_by_city(music_data)
        
        # Analysis
        analyze_correlations(demographics_df, music_data)
        predict_genre_from_demographics(demographics_df, music_data)
        predict_demographics_from_genre(demographics_df, music_data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()