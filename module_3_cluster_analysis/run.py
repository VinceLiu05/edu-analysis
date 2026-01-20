"""
Module 3: Cluster Analysis (3.3.3) - Main Run Script

Cluster Analysis

Execute complete clustering analysis workflow:
1. PCA dimensionality reduction
2. Determine optimal number of clusters (elbow method)
3. K-Means clustering
4. Generate visualizations
5. Analyze class-level cluster distribution
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pca_reduction import perform_pca_analysis
import warnings
from clustering import (
    find_optimal_clusters_elbow_only,
    perform_clustering,
    comprehensive_clustering_visualization_all_k,
    get_available_feature_columns,
    FEATURE_COLUMNS,
)

warnings.filterwarnings('ignore')

DEFAULT_MAX_K = 10
DEFAULT_VARIANCE_THRESHOLD = 0.8
DEFAULT_RANDOM_STATE = 42


def fill_missing_and_infinite_by_class_median(features_df, class_col='class_id', 
                                              skip_cols=['total_time_minutes', 'avg_qa_time_minutes'],
                                              verbose=True):
    """Fill missing and infinite values with class median."""
    df = features_df.copy()
    if class_col not in df.columns:
        raise ValueError(f"DataFrame must contain column: {class_col}")

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip_cols]
    if verbose:
        print(f"Processing {len(num_cols)} numeric columns (excluding {skip_cols})")
    
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    def _fill_group_median(group):
        for col in num_cols:
            if group[col].isna().any():
                median_val = group[col].median()
                if np.isnan(median_val):
                    median_val = df[col].median()
                group[col] = group[col].fillna(median_val)
        return group

    df = df.groupby(class_col, group_keys=False).apply(_fill_group_median)
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    if verbose:
        missing_after = df[num_cols].isna().sum().sum()
        print(f"Median filling complete, remaining missing values: {missing_after}")

    return df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Module 3: Cluster Analysis (3.3.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--features_file', required=True, 
                       help='Path to extracted features CSV file (from Module 2)')
    parser.add_argument('--output_folder', required=True,
                       help='Output folder for clustering results')
    parser.add_argument('--max_k', type=int, default=DEFAULT_MAX_K,
                       help=f'Maximum number of clusters for elbow method (default: {DEFAULT_MAX_K})')
    parser.add_argument('--variance_threshold', type=float, default=DEFAULT_VARIANCE_THRESHOLD,
                       help=f'PCA variance threshold (default: {DEFAULT_VARIANCE_THRESHOLD})')
    
    return parser.parse_args()


def main():
    """Execute complete cluster analysis workflow."""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.features_file):
        print(f"Error: Features file does not exist: {args.features_file}")
        sys.exit(1)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("=" * 50)
    print("Module 3: Cluster Analysis (3.3.3)")
    print("=" * 50)
    print(f"Features file:     {args.features_file}")
    print(f"Output folder:     {args.output_folder}")
    print(f"Max K:             {args.max_k}")
    print(f"Variance threshold: {args.variance_threshold}")
    print("=" * 50)
    
    # Load features
    print("\n[1/5] Loading features...")
    features_df = pd.read_csv(args.features_file, encoding='utf-8-sig')
    print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} columns")
    
    # Fill missing values
    print("\n[2/5] Processing missing and infinite values...")
    features_df = fill_missing_and_infinite_by_class_median(features_df)
    
    # Get feature columns
    feature_columns = get_available_feature_columns(features_df, FEATURE_COLUMNS)
    if not feature_columns:
        print("Error: No valid feature columns found")
        sys.exit(1)
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    if len(features_df) > 0:
        print("\nFeature Descriptive Statistics:")
        print(features_df[feature_columns].describe())
    
    # PCA analysis
    print("\n[3/5] Performing PCA dimensionality reduction...")
    _, X_pca, pca, _, feature_cols = perform_pca_analysis(features_df, args.output_folder)
    
    # Find optimal clusters
    print("\n[4/5] Finding optimal number of clusters (elbow method)...")
    optimal_k, X_cluster = find_optimal_clusters_elbow_only(
        X_pca, pca, max_k=args.max_k, 
        variance_threshold=args.variance_threshold,
        output_folder=args.output_folder
    )
    
    # Perform clustering
    print(f"\n[5/5] Performing K-Means clustering (K={optimal_k})...")
    _, features_df_clustered, _ = perform_clustering(
        X_cluster, optimal_k, features_df, args.output_folder
    )
    
    # Save clustered features
    clustered_file = os.path.join(args.output_folder, 'clustering_03_clustered_features.csv')
    features_df_clustered.to_csv(clustered_file, index=False, encoding='utf-8-sig')
    print(f"Clustered features saved to: {clustered_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    comprehensive_clustering_visualization_all_k(
        X_pca, features_df_clustered, feature_columns,
        ['cluster'], args.output_folder
    )
    
    # Class-level cluster distribution
    print("\nAnalyzing class-level cluster distribution...")
    if 'class_id' in features_df_clustered.columns and 'cluster' in features_df_clustered.columns:
        class_cluster_dist = (
            features_df_clustered
            .groupby(['class_id', 'cluster'])
            .size()
            .reset_index(name='count')
            .pivot(index='class_id', columns='cluster', values='count')
            .fillna(0)
            .astype(int)
        )
        
        class_cluster_dist_file = os.path.join(args.output_folder, 'clustering_04_class_distribution.csv')
        class_cluster_dist.to_csv(class_cluster_dist_file, encoding='utf-8-sig')
        print(f"Class-cluster distribution saved to: {class_cluster_dist_file}")
        
        print("\nClass-wise cluster distribution preview:")
        print(class_cluster_dist.head())
    else:
        print("Warning: 'class_id' or 'cluster' column not found — skipping class-level analysis.")
    
    # Summary
    print("\n" + "=" * 50)
    print("Cluster Analysis Complete!")
    print("=" * 50)
    print(f"Results saved to:  {args.output_folder}")
    print(f"Total samples:     {len(features_df_clustered):,}")
    print(f"Features:          {len(feature_cols)}")
    print(f"Optimal clusters:  {optimal_k}")
    print("=" * 50)
    
    return features_df_clustered


if __name__ == '__main__':
    main()

