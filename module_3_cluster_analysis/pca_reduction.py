import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

FEATURE_COLUMNS = [
    'qa_turns', 'avg_qa_time_minutes', 'avg_question_length',
    'copy_paste_score', 'answer_seeking_intensity', 'understanding_signal_strength',
    'course_progress_ratio', 'is_exam_week', 'day_period', 'is_in_class_time'
]

BINARY_COLUMNS = [
    'is_exam_week', 'is_in_class_time'
]

PLOT_CONFIG = {
    'figsize_normal': (8, 6),
    'figsize_large': (10, 8),
    'dpi': 300,
    'alpha': 0.6,
    'scatter_size': 30
}

def _identify_feature_types(available_cols):
    """Identify binary and continuous columns present in the data."""
    binary_cols_present = []
    continuous_cols = [col for col in available_cols if col not in binary_cols_present]
    return binary_cols_present, continuous_cols


def _handle_missing_and_infinite_values(X):
    """Handle missing and infinite values."""
    print("Processing missing and infinite values...")
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].dtype.kind in 'iufc':
            median_val = X[col].median(skipna=True)
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna("missing")

    print(f"Remaining NaN count after fill: {X.isna().sum().sum()}")
    return X


def _add_noise_to_constant_columns(X, continuous_cols):
    """Add small noise to constant continuous columns to avoid scaling issues."""
    if not continuous_cols:
        return X
    
    constant_cols = X[continuous_cols].columns[X[continuous_cols].var() == 0].tolist()
    if constant_cols:
        for col in constant_cols:
            X[col] += np.random.normal(0, 1e-8, len(X))
    
    return X


def _prepare_scaled_data(X, binary_cols_present, continuous_cols, available_cols):
    """Standardize continuous features and combine with binary features."""
    print("Standardizing continuous features...")
    scaler = StandardScaler()

    if continuous_cols:
        X_cont_scaled = scaler.fit_transform(X[continuous_cols])
        X_scaled_df = pd.DataFrame(X_cont_scaled, columns=continuous_cols, index=X.index)
    else:
        X_scaled_df = pd.DataFrame(index=X.index)

    for col in binary_cols_present:
        X_scaled_df[col] = X[col].astype(float)

    X_scaled_df = X_scaled_df.reindex(columns=available_cols)

    X_scaled = X_scaled_df.values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_scaled, scaler


def _perform_pca_computation(X_scaled):
    """Perform PCA computation and return results."""
    print("Performing PCA...")
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca


def _print_pca_summary(pca, used_columns):
    """Print PCA analysis summary with dynamic feature list."""
    print("\n=== PCA Analysis Results ===")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:6]):
        print(f"PC{i+1}: {ratio:.4f}")
    cumulative_variance = pca.explained_variance_ratio_[:3].sum()
    print(f"Cumulative (first 3): {cumulative_variance:.4f}")

    components_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=used_columns
    )
    print("\nFeature loadings for first 3 components:")
    print(components_df.round(3))
    return components_df


def _save_explained_variance_plot(pca, output_folder):
    """Save explained variance ratio plot."""
    plt.figure(figsize=PLOT_CONFIG['figsize_normal'])
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, 
        color='skyblue', 
        edgecolor='black'
    )
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Component')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, 'pca_01_explained_variance_ratio.png')
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def _save_cumulative_variance_plot(pca, output_folder):
    """Save cumulative explained variance plot."""
    plt.figure(figsize=PLOT_CONFIG['figsize_normal'])
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_), 
        'bo-'
    )
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, 'pca_02_cumulative_explained_variance.png')
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def _save_feature_loadings_heatmap(components_df, output_folder):
    """Save feature loadings heatmap."""
    plt.figure(figsize=PLOT_CONFIG['figsize_large'])
    sns.heatmap(
        components_df.T, 
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        cbar_kws={'label': 'Loading'}
    )
    plt.title('Feature Loadings in Principal Components')
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, 'pca_03_feature_loadings_heatmap.png')
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def _save_2d_projection_plot(X_pca, pca, output_folder):
    """Save 2D PCA projection scatter plot."""
    plt.figure(figsize=PLOT_CONFIG['figsize_normal'])
    plt.scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        alpha=PLOT_CONFIG['alpha'], 
        s=PLOT_CONFIG['scatter_size'], 
        edgecolor='black'
    )
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.title('PCA 2D Projection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, 'pca_04_2d_projection.png')
    plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def _save_all_plots(X_pca, pca, components_df, output_folder):
    """Save all PCA visualization plots."""
    os.makedirs(output_folder, exist_ok=True)

    _save_explained_variance_plot(pca, output_folder)
    _save_cumulative_variance_plot(pca, output_folder)
    _save_feature_loadings_heatmap(components_df, output_folder)
    _save_2d_projection_plot(X_pca, pca, output_folder)

    print(f"\nSaved individual PCA plots to: {output_folder}")
    print("  - pca_01_explained_variance_ratio.png")
    print("  - pca_02_cumulative_explained_variance.png")
    print("  - pca_03_feature_loadings_heatmap.png")
    print("  - pca_04_2d_projection.png")

def perform_pca_analysis(features_df, output_folder):
    """Perform PCA analysis (binary 0/1 variables are not standardized)."""
    available_cols = [col for col in FEATURE_COLUMNS if col in features_df.columns]
    if not available_cols:
        raise ValueError("No available feature columns found in features_df based on FEATURE_COLUMNS.")

    binary_cols_present, continuous_cols = _identify_feature_types(available_cols)

    X = features_df[available_cols].copy()

    X = _handle_missing_and_infinite_values(X)
    X = _add_noise_to_constant_columns(X, continuous_cols)

    X_scaled, scaler = _prepare_scaled_data(X, binary_cols_present, continuous_cols, available_cols)

    X_pca, pca = _perform_pca_computation(X_scaled)

    components_df = _print_pca_summary(pca, available_cols)

    _save_all_plots(X_pca, pca, components_df, output_folder)

    return X_scaled, X_pca, pca, scaler, available_cols