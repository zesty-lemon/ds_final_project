"""
Music Genre Classification: Hip Hop vs R&B
==========================================
Predictive analysis using audio features with AUC-ROC and 5-fold cross-validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# Configuration
TRANSLATED_FOLDER = Path('/Users/albinmeli/CS5870/ds_final_project/selected_cities_apple/translated')
OUTPUT_FOLDER = Path('/Users/albinmeli/CS5870/ds_final_project/outputs/genre_classification')
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

# Standardize genre variations
GENRE_MAPPINGS = {
    'hip hop': ['hip hop', 'hiphop', 'hip-hop'],
    'r&b': ['r&b', 'rnb', 'r and b', 'r & b', 'rhythm and blues'],
    'pop': ['pop', 'pop music'],
    'rock': ['rock', 'rock music'],
    'country': ['country', 'country music'],
    'latin': ['latin', 'latin music', 'latino'],
    'dance': ['dance', 'dance music', 'edm', 'electronic dance'],
    'alternative': ['alternative', 'alt', 'alternative rock'],
    'indie': ['indie', 'indie rock', 'indie pop'],
    'electronic': ['electronic', 'electro', 'electronica']
}

COLORS = {'hip_hop': '#FF6B6B', 'rnb': '#4ECDC4', 'primary': '#6C5CE7'}

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING
# ============================================================================

def standardize_genre(genre_normalized):
    """Map genre variations to canonical names."""
    for canonical, variations in GENRE_MAPPINGS.items():
        if genre_normalized in variations:
            return canonical
    return genre_normalized

def load_all_translated_data():
    """Load and combine all city music data."""
    print("\nLoading translated data...")
    
    all_data = []
    
    if not TRANSLATED_FOLDER.exists():
        print(f"Translated folder not found: {TRANSLATED_FOLDER}")
        return pd.DataFrame()
    
    # Load all Excel files
    excel_files = list(TRANSLATED_FOLDER.glob('*.xlsx')) or list(TRANSLATED_FOLDER.glob('*.xls'))
    
    if not excel_files:
        print(f"No Excel files found in {TRANSLATED_FOLDER}")
        return pd.DataFrame()
    
    print(f"Found {len(excel_files)} Excel files")
    
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file)
            city_name = excel_file.stem.replace('_translated', '')
            df['city'] = city_name
            all_data.append(df)
            print(f"Loaded {len(df)} songs from {city_name}")
        except Exception as e:
            print(f"Error loading {excel_file.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} songs from {len(excel_files)} cities")
    
    # Standardize genres
    if 'genre' in combined_df.columns:
        combined_df['genre_original'] = combined_df['genre']
        combined_df['genre_normalized'] = (
            combined_df['genre']
            .str.lower()
            .str.strip()
            .str.replace('-', ' ', regex=False)
            .str.replace(r'\s+', ' ', regex=True)
        )
        combined_df['genre_standardized'] = combined_df['genre_normalized'].apply(standardize_genre)
        
        print("\nTop 15 genres after standardization:")
        genre_counts = combined_df['genre_standardized'].value_counts().head(15)
        for i, (genre, count) in enumerate(genre_counts.items(), 1):
            print(f"{i:2d}. {genre:25s} : {count:5d} songs")
    
    return combined_df

def prepare_hip_hop_vs_rnb_data(df):
    """Filter and prepare Hip Hop vs R&B dataset."""
    print("\nPreparing Hip Hop vs R&B dataset...")
    
    if 'genre_standardized' not in df.columns:
        print("'genre_standardized' column not found")
        return None, None, None
    
    # Filter for Hip Hop and R&B only
    is_hip_hop = df['genre_standardized'] == 'hip hop'
    is_rnb = df['genre_standardized'] == 'r&b'
    df_filtered = df[is_hip_hop | is_rnb].copy()
    
    if len(df_filtered) == 0:
        print("No hip hop or r&b songs found")
        return None, None, None
    
    df_filtered['is_hip_hop'] = is_hip_hop[is_hip_hop | is_rnb].astype(int)
    
    # Check for missing features
    missing_features = [f for f in AUDIO_FEATURES if f not in df_filtered.columns]
    if missing_features:
        print(f"Missing audio features: {missing_features}")
        return None, None, None
    
    df_filtered = df_filtered.dropna(subset=AUDIO_FEATURES)
    
    print(f"Dataset prepared:")
    print(f"  Hip Hop songs: {df_filtered['is_hip_hop'].sum()}")
    print(f"  R&B songs: {(~df_filtered['is_hip_hop'].astype(bool)).sum()}")
    print(f"  Total: {len(df_filtered)} songs")
    
    X = df_filtered[AUDIO_FEATURES].values
    Y = df_filtered['is_hip_hop'].values
    
    return X, Y, df_filtered

# ============================================================================
# EXPLORATORY VISUALIZATIONS
# ============================================================================

def explore_features(df_filtered):
    """Create exploratory visualizations of audio features."""
    print("\nExploring audio features...")
    
    # Feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        
        hip_hop_data = df_filtered[df_filtered['is_hip_hop'] == 1][feature]
        rnb_data = df_filtered[df_filtered['is_hip_hop'] == 0][feature]
        
        ax.hist(hip_hop_data, bins=30, alpha=0.7, label='Hip Hop', 
                color=COLORS['hip_hop'], edgecolor='white', linewidth=1.5)
        ax.hist(rnb_data, bins=30, alpha=0.7, label='R&B', 
                color=COLORS['rnb'], edgecolor='white', linewidth=1.5)
        
        ax.set_xlabel(feature.capitalize(), fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature.capitalize()} Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
    
    if len(AUDIO_FEATURES) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Audio Feature Distributions: Hip Hop vs R&B', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '01_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: 01_feature_distributions.png")
    plt.close()
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    corr_matrix = df_filtered[AUDIO_FEATURES].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                square=True, ax=ax, linewidths=2, linecolor='white')
    
    ax.set_title('Audio Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '02_feature_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved: 02_feature_correlation.png")
    plt.close()
    
    # Feature importance (mean difference)
    mean_diff = {}
    for feature in AUDIO_FEATURES:
        hip_hop_mean = df_filtered[df_filtered['is_hip_hop'] == 1][feature].mean()
        rnb_mean = df_filtered[df_filtered['is_hip_hop'] == 0][feature].mean()
        mean_diff[feature] = abs(hip_hop_mean - rnb_mean)
    
    mean_diff_df = pd.DataFrame(list(mean_diff.items()), columns=['Feature', 'Mean Difference'])
    mean_diff_df = mean_diff_df.sort_values('Mean Difference', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(mean_diff_df['Feature'], mean_diff_df['Mean Difference'], 
                   color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Absolute Mean Difference', fontsize=13, fontweight='bold')
    ax.set_ylabel('Audio Feature', fontsize=13, fontweight='bold')
    ax.set_title('Feature Discriminative Power (Hip Hop vs R&B)', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(mean_diff_df.iterrows()):
        ax.text(row['Mean Difference'] + 0.01, i, f"{row['Mean Difference']:.3f}",
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '03_feature_importance_basic.png', dpi=300, bbox_inches='tight')
    print("Saved: 03_feature_importance_basic.png")
    plt.close()

# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate_models(X, Y):
    """Train models with 5-fold cross-validation."""
    print("\nTraining models with 5-fold cross-validation...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Logistic Regression': LogisticRegression(penalty=None, max_iter=1000, random_state=42),
        'Logistic Regression (L2)': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        
        fold_results = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [],
            'fpr': [], 'tpr': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, Y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            
            fold_results['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_results['precision'].append(precision_score(y_test, y_pred))
            fold_results['recall'].append(recall_score(y_test, y_pred))
            fold_results['f1'].append(f1_score(y_test, y_pred))
            fold_results['auc'].append(roc_auc_score(y_test, y_score))
            
            fpr, tpr, _ = roc_curve(y_test, y_score)
            fold_results['fpr'].append(fpr)
            fold_results['tpr'].append(tpr)
            
            print(f"  Fold {fold_idx}: Accuracy={fold_results['accuracy'][-1]:.3f}, "
                  f"AUC={fold_results['auc'][-1]:.3f}")
        
        print(f"\n  Average Metrics:")
        print(f"    Accuracy:  {np.mean(fold_results['accuracy']):.3f} ± {np.std(fold_results['accuracy']):.3f}")
        print(f"    Precision: {np.mean(fold_results['precision']):.3f} ± {np.std(fold_results['precision']):.3f}")
        print(f"    Recall:    {np.mean(fold_results['recall']):.3f} ± {np.std(fold_results['recall']):.3f}")
        print(f"    F1 Score:  {np.mean(fold_results['f1']):.3f} ± {np.std(fold_results['f1']):.3f}")
        print(f"    AUC-ROC:   {np.mean(fold_results['auc']):.3f} ± {np.std(fold_results['auc']):.3f}")
        
        results[model_name] = fold_results
    
    return results, scaler

def plot_roc_curves(results):
    """Plot ROC curves for all models."""
    print("\nGenerating ROC curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for idx, (model_name, fold_results) in enumerate(results.items()):
        ax = axes[idx]
        
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        
        # Plot individual fold ROC curves
        for fold_idx in range(len(fold_results['fpr'])):
            fpr = fold_results['fpr'][fold_idx]
            tpr = fold_results['tpr'][fold_idx]
            auc_score = fold_results['auc'][fold_idx]
            
            ax.plot(fpr, tpr, alpha=0.3, lw=2, 
                   label=f'Fold {fold_idx+1} (AUC={auc_score:.2f})',
                   color=plt.cm.Set3(fold_idx))
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        
        # Plot mean ROC curve
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(fold_results['auc'])
        std_auc = np.std(fold_results['auc'])
        
        ax.plot(mean_fpr, mean_tpr, color=COLORS['primary'], lw=4,
                label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        
        # Confidence interval
        std_tpr = np.std(interp_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                        label='± 1 std. dev.')
        
        ax.plot([0, 1], [0, 1], 'r--', lw=3, alpha=0.6, label='Chance (AUC=0.50)')
        
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name}\n5-Fold Cross-Validation ROC', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    plt.suptitle('ROC Curves: Hip Hop vs R&B Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '04_roc_curves_all_models.png', dpi=300, bbox_inches='tight')
    print("Saved: 04_roc_curves_all_models.png")
    plt.close()

def plot_metrics_comparison(results):
    """Create comprehensive metrics comparison."""
    print("\nGenerating metrics comparison...")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    
    comparison_data = []
    for model_name, fold_results in results.items():
        for metric in metrics:
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': np.mean(fold_results[metric]),
                'Std': np.std(fold_results[metric])
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        metric_data = comparison_df[comparison_df['Metric'] == metric]
        x_pos = np.arange(len(metric_data))
        
        ax.bar(x_pos, metric_data['Mean'].values, 
               yerr=metric_data['Std'].values,
               alpha=0.85, capsize=8, color=COLORS['primary'],
               edgecolor='white', linewidth=2)
        
        ax.set_ylabel(label, fontsize=13, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_data['Model'].values, rotation=25, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.4, linewidth=2)
        
        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(metric_data['Mean'].values, metric_data['Std'].values)):
            ax.text(i, mean_val + std_val + 0.03, f'{mean_val:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.delaxes(axes[-1])
    
    plt.suptitle('Model Performance Comparison: Hip Hop vs R&B', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '05_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: 05_metrics_comparison.png")
    plt.close()
    
    # Save summary
    summary_df = comparison_df.pivot(index='Model', columns='Metric', values=['Mean', 'Std'])
    summary_df.to_csv(OUTPUT_FOLDER / 'model_performance_summary.csv')
    print("Saved: model_performance_summary.csv")

# ============================================================================
# GENRE PAIR CLASSIFICATION
# ============================================================================

def classify_all_genre_pairs(df, best_model_config, top_n=12):
    """Classify all pairs of top N genres."""
    print(f"\nClassifying all pairs of top {top_n} genres...")
    
    if 'genre_standardized' not in df.columns:
        print("'genre_standardized' column not found")
        return None
    
    print(f"\nUsing best model: {best_model_config['name']}")
    print(f"   Mean AUC-ROC: {best_model_config['auc']:.3f}")
    
    top_genres = df['genre_standardized'].value_counts().head(top_n).index.tolist()
    print(f"\nTop {top_n} genres:")
    for i, genre in enumerate(top_genres, 1):
        count = (df['genre_standardized'] == genre).sum()
        print(f"  {i}. {genre}: {count} songs")
    
    genre_pairs = list(combinations(top_genres, 2))
    print(f"\nEvaluating {len(genre_pairs)} genre pairs...")
    
    auc_matrix = pd.DataFrame(index=top_genres, columns=top_genres, dtype=float)
    
    model = best_model_config['model']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    for pair_idx, (genre1, genre2) in enumerate(genre_pairs, 1):
        print(f"\n[{pair_idx}/{len(genre_pairs)}] Evaluating: {genre1} vs {genre2}")
        
        # Filter data for this genre pair
        is_genre1 = df['genre_standardized'] == genre1
        is_genre2 = df['genre_standardized'] == genre2
        df_pair = df[is_genre1 | is_genre2].copy()
        
        # Check for missing features
        missing_features = [f for f in AUDIO_FEATURES if f not in df_pair.columns]
        if missing_features:
            print(f"  Skipping: missing features")
            continue
        
        df_pair_clean = df_pair.dropna(subset=AUDIO_FEATURES)
        
        if len(df_pair_clean) < 20:
            print(f"  Skipping: insufficient data ({len(df_pair_clean)} songs)")
            continue
        
        Y = (df_pair_clean['genre_standardized'] == genre1).astype(int).values
        X = df_pair_clean[AUDIO_FEATURES].values
        
        genre1_count = Y.sum()
        genre2_count = len(Y) - genre1_count
        
        if genre1_count < 5 or genre2_count < 5:
            print(f"  Skipping: insufficient samples")
            continue
        
        print(f"  Dataset: {genre1_count} {genre1}, {genre2_count} {genre2}")
        
        # Cross-validate
        aucs = []
        try:
            for train_idx, test_idx in cv.split(X, Y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = Y[train_idx], Y[test_idx]
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_score = model.predict_proba(X_test_scaled)[:, 1]
                
                aucs.append(roc_auc_score(y_test, y_score))
            
            mean_auc = np.mean(aucs)
            print(f"  Mean AUC: {mean_auc:.3f} ± {np.std(aucs):.3f}")
            
            auc_matrix.loc[genre1, genre2] = mean_auc
            auc_matrix.loc[genre2, genre1] = mean_auc
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Fill diagonal and missing values
    np.fill_diagonal(auc_matrix.values, 1.0)
    auc_matrix = auc_matrix.fillna(0.5)
    
    auc_matrix.to_csv(OUTPUT_FOLDER / 'genre_pair_auc_matrix.csv')
    print(f"\nSaved AUC matrix to: genre_pair_auc_matrix.csv")
    
    plot_genre_pair_heatmap(auc_matrix, top_n)
    plot_genre_pair_clustermap(auc_matrix, top_n)
    
    return auc_matrix

def plot_genre_pair_heatmap(auc_matrix, top_n):
    """Plot heatmap of genre pair AUC scores."""
    print("\nGenerating genre pair AUC heatmap...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(auc_matrix.astype(float), annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0.75, vmin=0.5, vmax=1.0,
                square=True, ax=ax, linewidths=2, linecolor='white',
                cbar_kws={'label': 'Mean AUC-ROC'},
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    
    ax.set_title(f'Mean ROC AUC for Top {top_n} Genre Pair Classification\n(5-Fold Cross-Validation)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Genre', fontsize=13, fontweight='bold')
    ax.set_ylabel('Genre', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / '06_genre_pair_auc_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: 06_genre_pair_auc_heatmap.png")
    plt.close()

def plot_genre_pair_clustermap(auc_matrix, top_n):
    """Plot hierarchical clustering of genres by classification difficulty."""
    print("\nGenerating genre pair AUC clustermap...")
    
    g = sns.clustermap(auc_matrix.astype(float), annot=True, fmt='.2f',
                       cmap='RdYlGn', center=0.75, vmin=0.5, vmax=1.0,
                       figsize=(16, 14), linewidths=2, linecolor='white',
                       cbar_kws={'label': 'Mean AUC-ROC'},
                       annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    
    g.fig.suptitle(f'Hierarchical Clustering of Top {top_n} Genres by Classification Difficulty', 
                   fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_FOLDER / '07_genre_pair_auc_clustermap.png', dpi=300, bbox_inches='tight')
    print("Saved: 07_genre_pair_auc_clustermap.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\nMusic Genre Classification Analysis")
    print("Using Audio Features for Predictive Analysis")
    
    # Load data
    df = load_all_translated_data()
    
    if df.empty:
        print("\nNo data to analyze")
        return
    
    # Part 1: Hip Hop vs R&B
    print("\n" + "="*80)
    print("PART 1: HIP HOP VS R&B CLASSIFICATION")
    print("="*80)
    
    X, Y, df_filtered = prepare_hip_hop_vs_rnb_data(df)
    
    best_model_config = None
    
    if X is not None and Y is not None:
        explore_features(df_filtered)
        results, scaler = train_and_evaluate_models(X, Y)
        plot_roc_curves(results)
        plot_metrics_comparison(results)
        
        # Determine best model
        best_model_tuple = max(results.items(), key=lambda x: np.mean(x[1]['auc']))
        best_name = best_model_tuple[0]
        best_auc = np.mean(best_model_tuple[1]['auc'])
        best_acc = np.mean(best_model_tuple[1]['accuracy'])
        
        print(f"\nBest Model: {best_name}")
        print(f"  Mean AUC-ROC: {best_auc:.3f}")
        print(f"  Mean Accuracy: {best_acc:.3f}")
        
        # Configure best model for Part 2
        if best_name == 'Logistic Regression':
            best_model_instance = LogisticRegression(penalty=None, max_iter=1000, random_state=42)
        else:
            best_model_instance = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
        
        best_model_config = {
            'name': best_name,
            'model': best_model_instance,
            'auc': best_auc,
            'accuracy': best_acc
        }
    else:
        print("\nSkipping Part 1 due to data issues")
        best_model_config = {
            'name': 'Logistic Regression (Default)',
            'model': LogisticRegression(penalty=None, max_iter=1000, random_state=42),
            'auc': 0.0,
            'accuracy': 0.0
        }
    
    # Part 2: All Genre Pairs
    print("\n" + "="*80)
    print("PART 2: TOP 12 GENRE PAIR CLASSIFICATION")
    print("="*80)

    if best_model_config is not None:
        auc_matrix = classify_all_genre_pairs(df, best_model_config, top_n=12)
        
        if auc_matrix is not None:
            # Extract and rank pairs
            pairs_data = []
            genres = auc_matrix.index.tolist()
            for i, genre1 in enumerate(genres):
                for j, genre2 in enumerate(genres):
                    if i < j:
                        auc_val = auc_matrix.loc[genre1, genre2]
                        if not pd.isna(auc_val) and auc_val != 0.5:
                            pairs_data.append({
                                'Genre 1': genre1,
                                'Genre 2': genre2,
                                'Mean AUC': auc_val
                            })
            
            if pairs_data:
                pairs_df = pd.DataFrame(pairs_data).sort_values('Mean AUC', ascending=False)
                
                print("\nMost Easily Discriminable Genre Pairs (Top 5):")
                for idx, row in pairs_df.head(5).iterrows():
                    print(f"  {row['Genre 1'].title()} vs {row['Genre 2'].title()}: AUC = {row['Mean AUC']:.3f}")
                
                print("\nMost Difficult to Discriminate Genre Pairs (Bottom 5):")
                for idx, row in pairs_df.tail(5).iterrows():
                    print(f"  {row['Genre 1'].title()} vs {row['Genre 2'].title()}: AUC = {row['Mean AUC']:.3f}")
                
                pairs_df.to_csv(OUTPUT_FOLDER / 'genre_pairs_ranked.csv', index=False)
                print(f"\nSaved genre pairs ranking to: genre_pairs_ranked.csv")
                
                print(f"\nOverall Statistics:")
                print(f"  Total pairs evaluated: {len(pairs_df)}")
                print(f"  Mean AUC: {pairs_df['Mean AUC'].mean():.3f}")
                print(f"  Median AUC: {pairs_df['Mean AUC'].median():.3f}")
                print(f"  Min AUC: {pairs_df['Mean AUC'].min():.3f}")
                print(f"  Max AUC: {pairs_df['Mean AUC'].max():.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()