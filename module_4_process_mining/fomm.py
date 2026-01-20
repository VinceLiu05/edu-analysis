"""
Transition Probability Matrix (TPM) Heatmap Visualization
Analyzes cluster transition patterns across students
"""

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# Global State Definitions
# -------------------------------------------------------------
STATES = ["START", "c0", "c1", "c2", "c3", "END"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Cluster name mapping
CLUSTER_NAMES = {
    "START": "START",
    "END": "END",
    "c0": "Deep\nEngagement",
    "c1": "Shallow\nEngagement",
    "c2": "Routine-Learning\nEngagement",
    "c3": "Exam-Driven\nEngagement"
}

# Heatmap labels (no line breaks, suitable for heatmap display)
HEATMAP_LABELS = [
    "START",
    "Deep Engagement",
    "Shallow Engagement",
    "Routine-Learning Engagement", 
    "Exam-Driven Engagement",
    "END"
]

# -------------------------------------------------------------
# Core Functions
# -------------------------------------------------------------
def build_student_sequence(df_student):
    """
    Build dialogue sequence for a student.
    Note: df_student should contain all dialogues for one student in one class.
    """
    if df_student is None or df_student.empty:
        return ["START", "END"]

    df_student = df_student.copy()
    df_student["dialog_time"] = pd.to_datetime(df_student["dialog_time"], errors="coerce")
    df_student = df_student.sort_values("dialog_time")

    seq_mid = [f"c{int(c)}" for c in df_student["cluster"].values]
    return ["START"] + seq_mid + ["END"]


def build_transition_count_matrix(sequences):
    """
    Count transitions across all sequences, returns 6x6 matrix.
    """
    n = len(STATES)
    count_matrix = np.zeros((n, n), dtype=int)
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_s = seq[i]
            to_s = seq[i+1]
            
            if from_s in STATE_IDX and to_s in STATE_IDX:
                r = STATE_IDX[from_s]
                c = STATE_IDX[to_s]
                count_matrix[r, c] += 1
    
    return count_matrix


def normalize_to_prob(count_matrix):
    """
    Row-normalize count matrix to probability matrix.
    """
    prob_matrix = count_matrix.astype(float)
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    prob_matrix = prob_matrix / row_sums
    
    return prob_matrix


def plot_transition_heatmap(prob_matrix, out_path, title_suffix=""):
    """
    Plot transition probability heatmap.
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    
    # Create wrapped labels
    heatmap_labels_wrapped = [
        "START",
        "Deep\nEngagement",
        "Shallow\nEngagement",
        "Routine-Learning\nEngagement", 
        "Exam-Driven\nEngagement",
        "END"
    ]
    
    # Plot heatmap
    heatmap = sns.heatmap(prob_matrix, annot=False, cmap='BuGn',
                xticklabels=heatmap_labels_wrapped, yticklabels=heatmap_labels_wrapped,
                cbar_kws={'label': 'Transition Probability'},
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    # Manually add text annotations with color based on value
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            val = prob_matrix[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{val:.2f}',
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color=text_color)
    
    # Set colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Transition Probability', fontsize=18, labelpad=15)
    cbar.ax.tick_params(labelsize=16)
    
    # Set tick label font size, rotate x-axis labels 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='bold')
    
    ax.set_xlabel('To State', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('From State', fontsize=20, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved transition probability heatmap: {out_path}")


def plot_transition_heatmap_with_counts(prob_matrix, count_matrix, out_path, title_suffix=""):
    """
    Plot transition probability heatmap with count information.
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    
    # Create wrapped labels
    heatmap_labels_wrapped = [
        "START",
        "Deep\nEngagement",
        "Shallow\nEngagement",
        "Routine-Learning\nEngagement", 
        "Exam-Driven\nEngagement",
        "END"
    ]
    
    # Create custom annotation text: probability + (count)
    annot_text = np.empty_like(prob_matrix, dtype=object)
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            prob = prob_matrix[i, j]
            count = int(count_matrix[i, j])
            annot_text[i, j] = f"{prob:.2f}\n({count})"
    
    # Plot heatmap
    heatmap = sns.heatmap(prob_matrix, annot=False, cmap='BuGn',
                xticklabels=heatmap_labels_wrapped, yticklabels=heatmap_labels_wrapped,
                cbar_kws={'label': 'Transition Probability'},
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    # Manually add text annotations with color based on value
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            val = prob_matrix[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, annot_text[i, j],
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color=text_color)
    
    # Set colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Transition Probability', fontsize=18, labelpad=15)
    cbar.ax.tick_params(labelsize=18)
    
    # Set tick label font size, rotate x-axis labels 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='bold')
    
    ax.set_xlabel('To State', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel('From State', fontsize=20, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved transition probability heatmap (with counts): {out_path}")


def save_transition_matrices(count_matrix, prob_matrix, out_dir):
    """
    Save transition count matrix and probability matrix to CSV files.
    """
    # Save count matrix
    df_count = pd.DataFrame(count_matrix, 
                           index=HEATMAP_LABELS, 
                           columns=HEATMAP_LABELS)
    count_path = os.path.join(out_dir, "transition_count_matrix.csv")
    df_count.to_csv(count_path)
    print(f"  Saved transition count matrix: {count_path}")
    
    # Save probability matrix
    df_prob = pd.DataFrame(prob_matrix, 
                          index=HEATMAP_LABELS, 
                          columns=HEATMAP_LABELS)
    prob_path = os.path.join(out_dir, "transition_probability_matrix.csv")
    df_prob.to_csv(prob_path)
    print(f"  Saved transition probability matrix: {prob_path}")


def main(args):
    """
    Main function: Transition Probability Matrix Analysis
    """
    print("=" * 60)
    print("Transition Probability Matrix Analysis")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    df_cluster = pd.read_csv(args.cluster_csv)
    
    # Ensure required columns exist
    required_cols = ['student_id', 'class_id', 'cluster', 'dialog_time']
    missing_cols = [col for col in required_cols if col not in df_cluster.columns]
    if missing_cols:
        raise ValueError(f"cluster_csv missing required columns: {missing_cols}")
    
    # 2. Determine student-class combinations
    print("\n[2/4] Determining student-class combinations...")
    all_student_class_pairs = df_cluster[["student_id", "class_id"]].astype(str).drop_duplicates()
    print(f"  Found {len(all_student_class_pairs)} student-class combinations")
    
    # 3. Build sequences
    print("\n[3/4] Building sequences...")
    print("  Note: Dialogues for the same student in different classes are counted separately")
    
    all_sequences = []
    
    for _, row in tqdm(all_student_class_pairs.iterrows(), 
                       total=len(all_student_class_pairs), 
                       desc="  Processing"):
        stu_id = str(row['student_id'])
        cls_id = str(row['class_id'])
        
        # Filter dialogues for this student in this class
        df_stu = df_cluster[(df_cluster['student_id'].astype(str) == stu_id) & 
                            (df_cluster['class_id'].astype(str) == cls_id)]
        seq = build_student_sequence(df_stu)
        
        # Only include sequences with actual dialogues (more than just START->END)
        if len(seq) > 2:
            all_sequences.append(seq)
    
    print(f"  Total sequences with dialogues: {len(all_sequences)}")
    
    # 4. Calculate transition matrix and save results
    print("\n[4/4] Calculating transition probability matrix and saving results...")
    
    count_matrix = build_transition_count_matrix(all_sequences)
    prob_matrix = normalize_to_prob(count_matrix)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    plot_transition_heatmap(prob_matrix, 
                           os.path.join(args.output_dir, "transition_probability_heatmap.png"),
                           title_suffix=f"n={len(all_sequences)}")
    
    plot_transition_heatmap_with_counts(prob_matrix, count_matrix,
                                        os.path.join(args.output_dir, "transition_probability_heatmap_with_counts.png"),
                                        title_suffix=f"n={len(all_sequences)}")
    
    save_transition_matrices(count_matrix, prob_matrix, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transition Probability Matrix Analysis")
    parser.add_argument("--cluster_csv", required=True, 
                       help="Path to cluster features CSV file (required columns: student_id, class_id, cluster, dialog_time)")
    parser.add_argument("--output_dir", default="./results/tpm_analysis", 
                       help="Output directory")
    
    args = parser.parse_args()
    main(args)
