# Educational Data Analysis Pipeline

A Python toolkit for analyzing student-AI interaction data in online education platforms. This repository implements a complete data analysis pipeline with four modules corresponding to the research paper sections.

## Overview

This project provides a modular data analysis pipeline for educational research:

1. **Module 1: Conversation Session Identification and Segmentation (3.3.1)**
   - Two-stage dialogue segmentation algorithm
   - Time threshold (15-minute inactivity) + LLM topic segmentation

2. **Module 2: Feature Engineering (3.3.2)**
   - Extract 10 features from segmented sessions
   - Behavioral, cognitive, and temporal engagement features

3. **Module 3: Cluster Analysis (3.3.3)**
   - PCA dimensionality reduction
   - K-Means clustering with elbow method
   - Comprehensive visualization

4. **Module 4: Process Mining (3.3.4)**
   - First-Order Markov Model (FOMM)
   - Student-level engagement pattern analysis

## Installation

### Requirements

- **Python 3.8+** (required by dependencies: pandas>=2.0.0, numpy>=1.24.0, scikit-learn>=1.3.0)
- See `requirements.txt` for full dependency list

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Module 1: Session Segmentation

```bash
cd module_1_session_segmentation
python two_stage_dialog_split.py \
    --input_folder /path/to/raw/dialog/csv \
    --output_folder /path/to/segmented/output \
    [--enable_llm] \
    [--api_key YOUR_API_KEY]
```

### Module 2: Feature Engineering

```bash
cd module_2_feature_engineering
python run.py \
    --dialog_folder /path/to/segmented/dialogs \
    --class_time_file /path/to/class_time_range_by_school.csv \
    --class_schedule_file /path/to/class_schedule.csv \
    --output_file extracted_features.csv \
    --output_folder /path/to/output
```

### Module 3: Cluster Analysis

```bash
cd module_3_cluster_analysis
python run.py \
    --features_file /path/to/extracted_features.csv \
    --output_folder /path/to/clustering_results \
    --max_k 10 \
    --variance_threshold 0.8
```

### Module 4: Process Mining

```bash
cd module_4_process_mining
python fomm.py \
    --cluster_csv /path/to/clustered_features.csv \
    --output_dir /path/to/process_mining_results
```

## Project Structure

```
data_analysis/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── FEATURE_ENGINEERING.md             # Feature definitions and formulas
├── .gitignore                         # Git ignore rules
├── module_1_session_segmentation/    # Module 1: Session segmentation
│   ├── __init__.py
│   └── two_stage_dialog_split.py
├── module_2_feature_engineering/     # Module 2: Feature extraction
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── feature_utils.py
│   ├── io_utils.py
│   └── run.py
├── module_3_cluster_analysis/        # Module 3: Clustering
│   ├── __init__.py
│   ├── clustering.py
│   ├── pca_reduction.py
│   └── run.py
└── module_4_process_mining/           # Module 4: Process mining
    ├── __init__.py
    └── fomm.py
```

## Module Documentation

Each module contains:
- `__init__.py`: Module-level documentation (visible when importing)
- Main entry point:
  - Module 1: `two_stage_dialog_split.py`
  - Module 2: `run.py`
  - Module 3: `run.py`
  - Module 4: `fomm.py`
- Additional utility files as needed

## Output Files

### Module 2 Output
- `extracted_features.csv`: Feature matrix with 10 features per session
- `histograms_before_log/`: Feature distribution plots before log transformation
- `histograms_after_log/`: Feature distribution plots after log transformation

### Module 3 Output

**PCA Analysis Files:**
- `pca_01_explained_variance_ratio.png`: Explained variance ratio by component
- `pca_02_cumulative_explained_variance.png`: Cumulative explained variance ratio
- `pca_03_feature_loadings_heatmap.png`: Feature loadings in principal components
- `pca_04_2d_projection.png`: 2D PCA projection scatter plot

**Clustering Analysis Files:**
- `clustering_01_elbow_method.png`: Elbow method for optimal cluster number
- `clustering_02_stability_report.txt`: Clustering stability report (ARI scores)
- `clustering_03_clustered_features.csv`: Features with cluster labels
- `clustering_04_class_distribution.csv`: Class-level cluster distribution
- `clustering_05_pca_visualization.png`: PCA space clustering visualization
- `clustering_06_cluster_size_distribution.png`: Cluster size distribution bar chart
- `clustering_07_qa_turns_distribution.png`: QA turns distribution by cluster
- `clustering_08_feature_heatmap.png`: Cluster feature means heatmap
- `clustering_09_course_progress_distribution.png`: Course progress distribution by cluster

### Module 4 Output
- `transition_probability_heatmap.png`: State transition probability matrix (heatmap)
- `transition_probability_heatmap_with_counts.png`: Transition probability matrix with count annotations
- `transition_count_matrix.csv`: Raw transition count matrix
- `transition_probability_matrix.csv`: Transition probability matrix (CSV format)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

