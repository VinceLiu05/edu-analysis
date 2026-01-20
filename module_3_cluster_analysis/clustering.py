"""Clustering analysis module for K-Means clustering."""

from typing import Dict, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from matplotlib import rcParams

DEFAULT_VARIANCE_THRESHOLD = 0.8
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10

FEATURE_COLUMNS = [
    'qa_turns', 'avg_qa_time_minutes', 'avg_question_length',
    'copy_paste_score', 'answer_seeking_intensity', 'understanding_signal_strength',
    'course_progress_ratio', 'is_exam_week', 'day_period', 'is_in_class_time'
]

BINARY_COLUMNS = [
    'is_exam_week', 'is_in_class_time'
]

# Font configuration: use multiple fallback fonts, matplotlib will automatically select the first available one
rcParams['font.sans-serif'] = ['Heiti SC', 'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'STHeiti', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def get_available_feature_columns(df: pd.DataFrame, preferred_order=None):
    if preferred_order is None:
        preferred_order = FEATURE_COLUMNS
    return [col for col in preferred_order if col in df.columns]

def get_cluster_color_map(n_clusters: int) -> Dict[int, tuple]:
    """Generate color map for clusters."""
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    return {i: cmap(i) for i in range(n_clusters)}


def find_elbow_slope_method(inertias: List[float], K_range: range) -> int:
    """Find elbow point using slope change method."""
    if len(inertias) < 3:
        return K_range[0]

    slopes = [abs((inertias[i+1] - inertias[i]) / (K_range[i+1] - K_range[i])) 
              for i in range(len(inertias) - 1)]
    slope_changes = [abs(slopes[i+1] - slopes[i]) for i in range(len(slopes) - 1)]

    if not slope_changes:
        return K_range[0]

    elbow_idx = np.argmax(slope_changes) + 1
    return K_range[min(elbow_idx, len(K_range)-1)]


def find_optimal_clusters_elbow_only(X_pca, pca, max_k=10, variance_threshold=0.8, output_folder=None):
    """Find optimal clusters using elbow method with adaptive PCA."""
    explained = np.array(pca.explained_variance_ratio_, dtype=float) if hasattr(pca, 'explained_variance_ratio_') else None
    if explained is None:
        raise ValueError("PCA object missing 'explained_variance_ratio_' attribute")

    total_variance = np.sum(explained)
    explained_ratio = explained / total_variance
    cumulative_variance = np.cumsum(explained_ratio)

    n_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1
    actual_variance = cumulative_variance[n_components_needed-1] if cumulative_variance[-1] >= variance_threshold else cumulative_variance[-1]
    X_cluster = X_pca[:, :n_components_needed]

    print(f"\n=== Clustering Configuration ===")
    print(f"Components: {n_components_needed}, Shape: {X_cluster.shape}, Variance: {actual_variance:.3f} ({actual_variance*100:.1f}%)")

    inertias = []
    K_range = range(2, min(max_k + 1, len(X_cluster)))
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=DEFAULT_RANDOM_STATE, n_init=DEFAULT_N_INIT)
        kmeans.fit(X_cluster)
        inertias.append(kmeans.inertia_)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

    final_k = find_elbow_slope_method(inertias, K_range)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    pc_range = range(1, len(explained_ratio) + 1)
    ax1.bar(pc_range, explained_ratio, alpha=0.6, label='Individual')
    ax1.plot(pc_range, cumulative_variance, 'ro-', linewidth=2, label='Cumulative')
    ax1.axhline(y=variance_threshold, color='red', linestyle='--', alpha=0.7, label=f'Target {variance_threshold*100:.0f}%')
    ax1.axvline(x=n_components_needed, color='green', linestyle='--', alpha=0.7, label=f'Selected PC={n_components_needed}')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'PCA Explained Variance\n({n_components_needed} PCs: {actual_variance*100:.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(K_range, inertias, 'bo-', linewidth=3, markersize=10, label='Inertia Curve')
    ax2.axvline(x=final_k, color='red', linestyle='--', linewidth=3, label=f'Selected K={final_k}')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Within-Cluster Sum of Squares')
    ax2.set_title(f'Elbow Method (Slope)\n({n_components_needed} PCs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if len(inertias) > 2:
        ax3 = fig.add_subplot(gs[1, 0])
        slopes = [abs((inertias[i+1] - inertias[i]) / (K_range[i+1] - K_range[i])) for i in range(len(inertias) - 1)]
        slope_changes = [abs(slopes[i+1] - slopes[i]) for i in range(len(slopes) - 1)]
        if slope_changes:
            ax3.plot(K_range[2:], slope_changes, 'go-', linewidth=2, markersize=6)
            ax3.axvline(x=final_k, color='red', linestyle='--', alpha=0.7, label=f'K={final_k}')
            ax3.set_xlabel('Number of Clusters (K)')
            ax3.set_ylabel('Slope Change')
            ax3.set_title('Slope Change Method')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = f"""PCA Summary:
• Components: {n_components_needed}/{len(explained_ratio)}
• Variance: {actual_variance*100:.1f}% (target: {variance_threshold*100:.0f}%)

Elbow Method:
• Selected K: {final_k}

Configuration:
• Algorithm: K-Means
• Feature space: {n_components_needed}D PCA"""
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    plt.suptitle(f'PCA + Elbow Method (Slope)\n{n_components_needed} PCs, {actual_variance*100:.1f}% → K={final_k}',
                 fontsize=14, y=0.98)
    plt.tight_layout()

    if output_folder:
        plt.savefig(os.path.join(output_folder, 'clustering_01_elbow_method.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return final_k, X_cluster

def check_clustering_stability(X_cluster, n_clusters, n_iterations=50, output_folder=None):
    """Check clustering stability by running K-Means multiple times and computing pairwise ARI."""
    print(f"\n=== Clustering Stability (K={n_clusters}, Iterations={n_iterations}) ===")
    
    seeds = range(DEFAULT_RANDOM_STATE, DEFAULT_RANDOM_STATE + n_iterations)
    labels_list = [KMeans(n_clusters=n_clusters, random_state=seed, n_init=DEFAULT_N_INIT)
                   .fit_predict(X_cluster) for seed in seeds]
    
    ari_scores = [adjusted_rand_score(labels_list[i], labels_list[j])
                  for i in range(n_iterations) for j in range(i + 1, n_iterations)]
    
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    min_ari = np.min(ari_scores)
    
    print(f"Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}, Min: {min_ari:.4f}")
    stability = "Highly Stable" if mean_ari > 0.8 else "Moderately Stable" if mean_ari > 0.5 else "Unstable"
    print(f"Result: {stability}")

    if output_folder:
        result_file = os.path.join(output_folder, 'clustering_02_stability_report.txt')
        with open(result_file, 'w') as f:
            f.write(f"Clustering Stability Report (K={n_clusters})\n{'='*40}\n")
            f.write(f"Iterations: {n_iterations}\nMean ARI: {mean_ari:.4f}\n")
            f.write(f"Std ARI: {std_ari:.4f}\nMin ARI: {min_ari:.4f}\n\nPairwise Scores:\n{ari_scores}\n")
        print(f"Report saved: {result_file}")
        
    return mean_ari

def perform_clustering(X_cluster, n_clusters, features_df, output_folder):
    """Perform clustering and save results."""
    os.makedirs(output_folder, exist_ok=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=DEFAULT_N_INIT)
    cluster_labels = kmeans.fit_predict(X_cluster)
    features_df_clustered = features_df.copy()
    features_df_clustered['cluster'] = cluster_labels

    print(f"\n=== Clustering Results (K={n_clusters}) ===")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cid, count in cluster_counts.items():
        print(f"Cluster {cid}: {count} samples ({count/len(cluster_labels)*100:.1f}%)")
    
    check_clustering_stability(X_cluster, n_clusters, output_folder=output_folder)

    return cluster_labels, features_df_clustered, kmeans


def comprehensive_clustering_visualization_all_k(
    X_pca, features_df_clustered, feature_columns, cluster_cols_list, output_folder
):
    """Generate comprehensive visualizations for clustering results."""
    os.makedirs(output_folder, exist_ok=True)

    for cluster_col in cluster_cols_list:
        if cluster_col not in features_df_clustered.columns:
            print(f"[Skipped] {cluster_col} not in DataFrame")
            continue

        cluster_labels = features_df_clustered[cluster_col].values
        n_clusters = len(np.unique(cluster_labels))
        colors_dict = get_cluster_color_map(n_clusters)
        colors = [colors_dict[label] for label in cluster_labels]
        print(f"Generating visualizations for {cluster_col} (K={n_clusters})...")

        # PCA 2D visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.title(f'PCA Clustering ({cluster_col})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        for i in range(n_clusters):
            cluster_points = X_pca[cluster_labels == i]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='x', linewidth=3)
                plt.annotate(f'C{i}', (centroid[0], centroid[1]), xytext=(5, 5), textcoords='offset points',
                             fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'clustering_05_pca_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Cluster size distribution
        plt.figure(figsize=(8, 6))
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        bars = plt.bar(range(n_clusters), cluster_counts.values,
                       color=[colors_dict[i] for i in range(n_clusters)], alpha=0.7, edgecolor='black')
        plt.title(f'Cluster Size Distribution ({cluster_col})')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Samples')
        plt.xticks(range(n_clusters))
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}',
                     ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'clustering_06_cluster_size_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # QA turns distribution
        plt.figure(figsize=(8, 6))
        qa_data, qa_labels, qa_stats = [], [], []
        for i in range(n_clusters):
            cluster_qa = features_df_clustered[features_df_clustered[cluster_col] == i]['qa_turns']
            if len(cluster_qa) > 0:
                orig_mean, orig_std = cluster_qa.mean(), cluster_qa.std()
                qa_min, qa_max = cluster_qa.min(), cluster_qa.max()
                cluster_qa_norm = (cluster_qa - qa_min) / (qa_max - qa_min) if qa_max > qa_min else pd.Series(0.5, index=cluster_qa.index)
                qa_data.append(cluster_qa_norm)
                qa_labels.append(f'C{i}')
                qa_stats.append((orig_mean, orig_std))
        if qa_data:
            box_plot = plt.boxplot(qa_data, labels=qa_labels, patch_artist=True)
            for patch, i in zip(box_plot['boxes'], range(len(qa_data))):
                patch.set_facecolor(colors_dict[i])
                patch.set_alpha(0.7)
            for i, (mean_val, std_val) in enumerate(qa_stats, start=1):
                plt.text(i, 1.05, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.title(f'QA Turns Distribution ({cluster_col})')
        plt.xlabel('Cluster Label')
        plt.ylabel('Normalized QA Turns (0–1)')
        plt.ylim(-0.05, 1.15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'clustering_07_qa_turns_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Feature heatmap
        available_cols = get_available_feature_columns(features_df_clustered, feature_columns)
        plot_cluster_feature_heatmap(features_df_clustered, available_cols, cluster_col,
                                     os.path.join(output_folder, 'clustering_08_feature_heatmap.png'))

        # Course progress distribution
        if 'course_progress_ratio' in features_df_clustered.columns:
            plt.figure(figsize=(8, 6))
            for i in range(n_clusters):
                cluster_weeks = features_df_clustered[features_df_clustered[cluster_col] == i]['course_progress_ratio']
                cluster_weeks = cluster_weeks[cluster_weeks > 0]
                if len(cluster_weeks) > 0:
                    plt.hist(cluster_weeks, bins=20, alpha=0.7, label=f'C{i}', color=colors_dict[i])
            plt.legend()
            plt.title(f'Course Progress Distribution ({cluster_col})')
            plt.xlabel('Course Progress Ratio')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'clustering_09_course_progress_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Saved visualizations for {cluster_cols_list} in {output_folder}")

def plot_cluster_feature_heatmap(features_df_clustered, feature_columns, cluster_col, output_file):
    """Plot cluster feature heatmap with Z-score normalization for numeric features."""
    available_cols = [col for col in feature_columns if col in features_df_clustered.columns]
    if not available_cols:
        raise ValueError("No available feature columns present.")

    cluster_means = features_df_clustered.groupby(cluster_col)[available_cols].mean()
    cluster_means_norm = cluster_means.copy()
    
    if available_cols:
        means = cluster_means[available_cols].mean()
        stds = cluster_means[available_cols].std().replace(0, 1e-8)
        cluster_means_norm[available_cols] = (cluster_means[available_cols] - means) / stds

    plt.figure(figsize=(10, max(6, len(available_cols)*0.3)))
    sns.heatmap(cluster_means_norm.T, cmap='RdBu_r', annot=False, cbar=True,
                xticklabels=[f'C{i}' for i in cluster_means.index],
                yticklabels=cluster_means_norm.columns)
    plt.title(f'Cluster Feature Heatmap ({cluster_col})')
    plt.xlabel('Cluster Label')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
