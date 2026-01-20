import os
import glob
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io_utils import load_reference_data
from feature_utils import (
    detect_has_image,
    calculate_copy_paste_score,
    calculate_answer_seeking_intensity,
    calculate_understanding_signal_strength
)

def compute_question_length(text):
    """Compute question text length after removing markdown images and HTML img tags."""
    if not isinstance(text, str):
        return 0

    s = text
    s = re.sub(r'!\[[^\]]*?\]\([^\)]*?\)', '', s)
    s = re.sub(r'请上传你的文件（Please upload the file）', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:Notice|注意)[:：].*?(?=$|\n)', '', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<img\b[^>]*>', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)
    return len(s.strip())

def analyze_nan_list(arr, name, nan_stats_dict):
    """Analyze NaN values in array and update stats dict."""
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return np.nan

    nan_count = np.sum(np.isnan(arr))
    if nan_count == len(arr):
        nan_stats_dict[f"{name}_all_nan"] += 1
    elif 0 < nan_count < len(arr):
        nan_stats_dict[f"{name}_partial_nan"] += 1

    return np.nanmean(arr) if nan_count < len(arr) else np.nan



def extract_features_from_dialog(
    file_path, df_class, df_schedule, df_school_bundle=None, stats=None
):
    """Extract features from a single dialog file."""
    try:
        if stats is not None:
            stats['total'] = stats.get('total', 0) + 1

        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            if stats is not None:
                stats['File does not exist'] = stats.get('File does not exist', 0) + 1
            return None

        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if df.empty:
            print(f"File is empty: {os.path.basename(file_path)}")
            if stats is not None:
                stats['File is empty'] = stats.get('File is empty', 0) + 1
            return None

        required_columns = ['提问时间', '提问内容', 'AI回复']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"File missing required columns {missing_columns}: {os.path.basename(file_path)}")
            if stats is not None:
                stats['missing columns'] = stats.get('missing columns', 0) + 1
            return None

        df.fillna("", inplace=True)
        file_name = os.path.basename(file_path)

        if '教学班ID' not in df.columns:
            print(f"Warning: File missing class ID column: {file_name}")
            if stats is not None:
                stats['failed'] = stats.get('failed', 0) + 1
            return None
        class_id = df["教学班ID"].iloc[0]

        if '学生ID' not in df.columns:
            print(f"Warning: File missing student ID column: {file_name}")
            if stats is not None:
                stats['missing student id'] = stats.get('missing student id', 0) + 1
            return None
        student_id = df["学生ID"].iloc[0]
        try:
            df["提问时间"] = pd.to_datetime(df["提问时间"], errors='coerce')
        except Exception as e:
            print(f"Time conversion failed: {file_name} - {e}")
            if stats is not None:
                stats['Time conversion'] = stats.get('Time conversion', 0) + 1
            return None

        if df["提问时间"].isna().any():
            print(f"Invalid QA times in: {file_name}")
            if stats is not None:
                stats['Invalid QA times'] = stats.get('Invalid QA times', 0) + 1
            return None
        
        if isinstance(df_school_bundle, dict):
            df_school = df_school_bundle.get('df_school', pd.DataFrame())
            df_class_info = df_school_bundle.get('df_class_info', pd.DataFrame())
        else:
            df_school = df_school_bundle
            df_class_info = pd.DataFrame()
        
        if df_school is not None and isinstance(df_school, dict) and 'df_class_info' in df_school:
            df_class_info = df_school['df_class_info']
            if not df_class_info.empty:
                valid_class_ids = set(df_class_info['教学班ID'].astype(str))
                if str(class_id) not in valid_class_ids:
                    print(f"Skipping class_id={class_id} (not in class_info_file): {file_name}")
                    if stats is not None:
                        stats['filtered_by_class_info'] = stats.get('filtered_by_class_info', 0) + 1
                    return None
        class_info = df_class[df_class['教学班ID'] == class_id]
        if class_info.empty:
            print(f"No class info for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['No class info'] = stats.get('No class info', 0) + 1
            return None

        course_start = pd.to_datetime(class_info['起始时间'].iloc[0], errors='coerce')
        course_end = pd.to_datetime(class_info['结束时间'].iloc[0], errors='coerce')
        if pd.isna(course_start) or pd.isna(course_end):
            print(f"Invalid course start/end time for class_id={class_id}: {file_name}")
            if stats is not None:
                stats['Invalid course start/end time'] = stats.get('Invalid course start/end time', 0) + 1
            return None

        qa_min_time = df["提问时间"].min()
        qa_max_time = df["提问时间"].max()
        out_of_range_early = qa_min_time < course_start
        out_of_range_late = qa_max_time > course_end
        if out_of_range_early or out_of_range_late:
            print(f"Dialog outside course window [{course_start}, {course_end}]: {file_name}")
            if stats is not None:
                stats['out_of_range'] = stats.get('out_of_range', 0) + 1
            return None

        qa_turns = len(df)
        if qa_turns > 1:
            total_time = max(0, (df["提问时间"].max() - df["提问时间"].min()).total_seconds() / 60)
        else:
            total_time = 0
        avg_qa_time = total_time / (qa_turns - 1) if qa_turns > 1 and total_time > 0 else 0

        question_lengths = df["提问内容"].apply(compute_question_length)
        
        if '提问入口' in df.columns:
            confusion_entries = {"课堂不懂", "课件不懂", "习题不懂"}
            mask = df.get("提问入口", pd.Series([], dtype=object)).fillna("").isin(confusion_entries)
            is_confusion_entry = int(mask.any()) if qa_turns > 0 else 0
        else:
            print(f"Warning: File missing entry point column: {file_name}")
            is_confusion_entry = 0
        
        has_image = detect_has_image(df["提问内容"])
        _, copy_paste_score = calculate_copy_paste_score(
            prompts=df["提问内容"], is_confusion_entry=is_confusion_entry
        )
        answer_seeking_intensity = calculate_answer_seeking_intensity(
            prompts=df["提问内容"],
            ai_responses=df["AI回复"] if "AI回复" in df.columns else pd.Series([]),
            has_image=has_image,
            is_confusion_entry=is_confusion_entry
        )
        understanding_signal_strength = calculate_understanding_signal_strength(
            prompts=df["提问内容"]
        )

        avg_question_length = float(question_lengths.mean()) if len(question_lengths) > 0 else 0.0

        qa_start_time = df["提问时间"].min()
        
        # Calculate course progress
        total_weeks = 20
        if pd.notna(course_start) and pd.notna(course_end) and course_end > course_start:
            days_total = (course_end.normalize() - course_start.normalize()).days
            days_per_week = days_total / total_weeks
            week_progress = int(((qa_start_time.normalize() - course_start.normalize()).days // days_per_week) + 1)
            week_progress = max(1, min(total_weeks, week_progress))
        else:
            week_progress = 1

        # Calculate time period features
        raw_day_period = qa_start_time.hour + qa_start_time.minute / 60.0
        shifted = (raw_day_period - 8.0) % 24.0
        day_period = int(shifted // 1) + 1

        # Calculate if exam week
        is_exam_week = 0
        if pd.notna(course_end):
            window_start = course_end - pd.Timedelta(days=28)
            is_exam_week = int(window_start <= qa_start_time <= course_end)

        # Calculate if class time
        def check_in_class_time(qa_time, class_id, df_schedule):
            try:
                schedule = df_schedule[df_schedule["教学班ID"].astype(str) == str(class_id)]
            except Exception:
                schedule = df_schedule[df_schedule["教学班ID"] == class_id]
            for _, row in schedule.iterrows():
                start = pd.to_datetime(row["开课时间"], errors="coerce")
                end = pd.to_datetime(row["结课时间"], errors="coerce")
                if pd.notna(start) and pd.notna(end) and start <= qa_time <= end:
                    return True
            return False

        is_in_class_time = int(any(check_in_class_time(t, class_id, df_schedule)
                                   for t in df["提问时间"] if pd.notna(t)))

        # Create complete feature dictionary at once
        features = {
            "file_name": file_name,
            "class_id": class_id,
            "student_id": student_id,
            "dialog_time": df["提问时间"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "qa_turns": int(qa_turns),
            "avg_qa_time_minutes": float(avg_qa_time),
            "avg_question_length": float(avg_question_length),
            "copy_paste_score": float(copy_paste_score),
            "answer_seeking_intensity": float(answer_seeking_intensity),
            "understanding_signal_strength": float(understanding_signal_strength),
            "course_progress_ratio": int(week_progress),
            "is_exam_week": int(is_exam_week),
            "day_period": int(day_period),
            "is_in_class_time": int(is_in_class_time),
        }
        for key, value in features.items():
            if key not in ['file_name', 'class_id', 'student_id', 'dialog_time','hours_to_next_class', 'hours_from_last_class']:
                if not np.isfinite(value) and not isinstance(value, (bool, str, int)):
                    features[key] = 0.0

        if stats is not None:
            stats['processed'] = stats.get('processed', 0) + 1

        return features

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_feature_histograms(df, features=None, bins=50, save_dir=None, figsize=(7, 4), stats_file="feature_stats.csv"):
    """Plot feature distributions and save statistics."""
    if save_dir is None:
        save_dir = os.path.abspath("histograms")
    os.makedirs(save_dir, exist_ok=True)

    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Auto-detected {len(features)} numeric features")

    saved_files = []
    stats_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Skipping missing feature: {feature}")
            continue

        series = pd.to_numeric(df[feature], errors='coerce').dropna()
        if series.empty:
            print(f"No data for feature: {feature}")
            continue

        stats = {
            "feature": feature,
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "median": series.median(),
            "variance": series.var()
        }
        stats_list.append(stats)

        plt.figure(figsize=figsize)
        unique_vals = sorted(series.unique())

        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            counts = series.value_counts().sort_index()
            plt.bar(counts.index.astype(str), counts.values, color="skyblue", edgecolor="black")
            plt.title(f"{feature} (binary: {unique_vals})")
            plt.xlabel(feature)
            plt.ylabel("Count")
        else:
            plt.hist(series, bins=bins, color="steelblue", edgecolor="black", alpha=0.75)
            plt.axvline(stats["min"], color='gray', linestyle='--', label=f"Min={stats['min']:.2f}")
            plt.axvline(stats["max"], color='gray', linestyle='--', label=f"Max={stats['max']:.2f}")
            plt.axvline(stats["mean"], color='red', linestyle='--', label=f"Mean={stats['mean']:.2f}")
            plt.axvline(stats["median"], color='green', linestyle=':', label=f"Median={stats['median']:.2f}")
            textstr = '\n'.join((f"Min: {stats['min']:.2f}", f"Max: {stats['max']:.2f}",
                                f"Mean: {stats['mean']:.2f}", f"Median: {stats['median']:.2f}",
                                f"Var: {stats['variance']:.2f}"))
            plt.text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.7))
            plt.legend()
            plt.title(f"{feature} (n={len(series)})")
            plt.xlabel(feature)
            plt.ylabel("Count")

        fname = os.path.join(save_dir, f"{feature}_dist.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        saved_files.append(fname)
        print(f"Saved plot: {fname}")

    stats_df = pd.DataFrame(stats_list)
    stats_path = os.path.join(save_dir, stats_file)
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved {len(saved_files)} plots and stats to: {stats_path}")

    return saved_files, stats_df

def adjust_single_turn_durations_from_list(features_list):
    """Adjust single-turn dialog avg_qa_time_minutes using median of 2-turn or 3-turn dialogs."""
    df = pd.DataFrame(features_list)
    df["qa_turns"] = pd.to_numeric(df.get("qa_turns", 0), errors="coerce").fillna(0).astype(int)
    
    # Only process avg_qa_time_minutes since total_time_minutes is no longer in features
    if "avg_qa_time_minutes" not in df.columns:
        return features_list
    
    df["avg_qa_time_minutes"] = pd.to_numeric(df["avg_qa_time_minutes"], errors="coerce").fillna(0)

    # Calculate medians by class for 2-turn dialogs
    medians_by_class = (
        df[df["qa_turns"] == 2]
        .groupby("class_id")[["avg_qa_time_minutes"]]
        .median()
        .rename(columns={"avg_qa_time_minutes": "class_2turn_avg_qa_time_minutes_median"})
    )

    # Calculate global medians for 2-turn and 3-turn dialogs
    global_2_median = df[df["qa_turns"] == 2]["avg_qa_time_minutes"].median()
    global_3_median = df[df["qa_turns"] == 3]["avg_qa_time_minutes"].median()

    if pd.isna(global_2_median) and pd.isna(global_3_median):
        print("Warning: No 2-turn or 3-turn dialogs found, skipping adjustment")
        return features_list

    # Merge class medians
    df = df.merge(medians_by_class, on="class_id", how="left")
    new_df = df.copy()
    mask_single = new_df["qa_turns"] == 1
    corrected_count = 0

    # Adjust single-turn dialogs
    for i, row in new_df.loc[mask_single].iterrows():
        class_median = row.get("class_2turn_avg_qa_time_minutes_median")
        if pd.notna(class_median):
            new_df.at[i, "avg_qa_time_minutes"] = class_median / 3
        elif pd.notna(global_2_median):
            new_df.at[i, "avg_qa_time_minutes"] = global_2_median / 3
        elif pd.notna(global_3_median):
            new_df.at[i, "avg_qa_time_minutes"] = global_3_median / 3
        corrected_count += 1

    print(f"Adjusted {corrected_count} single-turn dialog avg_qa_time_minutes")

    # Drop temporary columns
    drop_cols = [c for c in new_df.columns if c.startswith("class_2turn_")]
    new_df = new_df.drop(columns=drop_cols)
    return new_df.to_dict(orient="records")

def extract_all_features(
    dialog_folder, class_time_file, class_schedule_file, 
    school_info_file=None, class_info_file=None, final_week_file=None,
    plot_histograms=True, output_root=None
):
    """Extract features from all dialog files."""
    print("Loading reference data...")
    try:
        df_class, df_schedule, df_school = load_reference_data(
            class_time_file, 
            class_schedule_file, 
            school_info_file=school_info_file,
            class_info_file=class_info_file
        )
    except Exception as e:
        print(f"Failed to load reference data: {e}")
        df_class, df_schedule, df_school = pd.DataFrame(), pd.DataFrame(), {}

    print(f"Searching for dialog files in: {dialog_folder}")

    patterns = [
        os.path.join(dialog_folder, "*.csv"),
        os.path.join(dialog_folder, "*", "*.csv"),
        os.path.join(dialog_folder, "*", "*", "*.csv"),
        os.path.join(dialog_folder, "**", "*.csv"),
    ]
    csv_files = list(set(f for pattern in patterns for f in glob.glob(pattern, recursive=True)))

    exclude_keywords = ['feature', 'cluster', 'result', 'statistic', 'analysis', 'pca', 'failed']
    dialog_files = [f for f in csv_files 
                    if "cluster" not in f.replace("\\", "/") 
                    and not any(kw in os.path.basename(f).lower() for kw in exclude_keywords)]

    print(f"Found {len(dialog_files)} potential dialog CSV files")

    if not dialog_files:
        print("No CSV files found! Folder structure:")
        for root, _, files in os.walk(dialog_folder):
            level = root.replace(dialog_folder, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            csv_files_in_dir = [f for f in files if f.endswith('.csv')]
            for file in csv_files_in_dir[:5]:
                print(f'{subindent}{file}')
            if len(csv_files_in_dir) > 5:
                print(f'{subindent}... {len(csv_files_in_dir)-5} more CSV files')
        return pd.DataFrame()

    print("Validating file formats...")
    required_columns = ['提问时间', '提问内容', 'AI回复']
    valid_files = []
    
    if output_root:
        high_understanding_dir = os.path.join(output_root, "high_understanding_dialogs")
        os.makedirs(high_understanding_dir, exist_ok=True)
    
    for file_path in dialog_files[:10]:
        try:
            df_test = pd.read_csv(file_path, encoding='utf-8-sig', nrows=3)
            if all(col in df_test.columns for col in required_columns):
                valid_files.append(file_path)
                print(f"Valid file: {os.path.basename(file_path)}")
            else:
                print(f"Invalid file (missing columns): {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to read: {os.path.basename(file_path)} - {e}")
    
    df_final_week = None
    if final_week_file:
        try:
            df_final_week = pd.read_csv(final_week_file, encoding="utf-8-sig")
            print(f"Loaded final week info: {len(df_final_week)} rows")
        except Exception as e:
            print(f"Failed to load final week CSV: {final_week_file} - {e}")
    
    if len(valid_files) == min(10, len(dialog_files)):
        valid_files = dialog_files
        print(f"First 10 files valid, assuming all {len(dialog_files)} files are valid")
    else:
        print("Validating all files...")
        valid_files = [f for f in dialog_files 
                      if all(col in pd.read_csv(f, encoding='utf-8-sig', nrows=3).columns 
                             for col in required_columns)]

    print(f"Final number of valid files: {len(valid_files)}")

    if not valid_files:
        print("Error: No valid dialog files found!")
        return pd.DataFrame()

    features_list = []
    failed_count = 0
    stats = {'total': 0, 'processed': 0, 'failed': 0, 'out_of_range': 0}

    for i, file_path in enumerate(valid_files):
        if i % 100 == 0:
            print(f"Processing progress: {i+1}/{len(valid_files)}")

        features = extract_features_from_dialog(
            file_path, df_class, df_schedule,
            df_school_bundle=df_school, stats=stats
        )

        if features is not None:
            features_list.append(features)
            if output_root:
                understanding_score = features.get("understanding_signal_strength", 0.0)
                if understanding_score > 10:
                    try:
                        base_name = os.path.basename(file_path)
                        new_name = f"score_{understanding_score:.2f}_{base_name}"
                        shutil.copy(file_path, os.path.join(high_understanding_dir, new_name))
                    except Exception as e:
                        print(f"Failed to copy high understanding file {file_path}: {e}")
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"Feature extraction failed: {os.path.basename(file_path)}")
            if output_root:
                failed_dir = os.path.join(output_root, "failed")
                os.makedirs(failed_dir, exist_ok=True)
                try:
                    shutil.copy(file_path, os.path.join(failed_dir, os.path.basename(file_path)))
                except Exception as e:
                    print(f"Failed to copy to failed folder: {e}")

    print(f"Extracted features from {len(features_list)} dialogs, failed on {failed_count} files")

    total = int(stats.get('total', 0))
    processed = int(stats.get('processed', 0))
    out_of_range = int(stats.get('out_of_range', 0))
    failed_other = int(stats.get('failed', 0))

    stats_to_save = {
        "valid_files_found": int(len(valid_files)),
        "total_dialog_calls": total,
        "processed_dialogs": processed,
        "out_of_range_dialogs": out_of_range,
        "successful_feature_rows": int(len(features_list)),
        "stats": stats
    }
    if total > 0:
        stats_to_save.update({
            "success_ratio": processed / total,
            "out_of_range_ratio": out_of_range / total,
            "fail_ratio": (out_of_range + failed_other) / total
        })

    if output_root is None:
        output_root = os.path.join(dialog_folder, "clustering_results")
    os.makedirs(output_root, exist_ok=True)

    stats_path = os.path.join(output_root, "dialog_stats_miss.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
    print(f"Stats saved to: {stats_path}")

    if not features_list:
        print("Warning: No features were successfully extracted!")
        return pd.DataFrame()
    
    print("Adjusting single-turn dialog durations...")
    new_features_list = adjust_single_turn_durations_from_list(features_list)
    features_df = pd.DataFrame(new_features_list)

    if "total_time_minutes" in features_df.columns:
        features_df = features_df.drop(columns=["total_time_minutes"])

    if plot_histograms:
        hist_dir = os.path.join(output_root, 'histograms_before_log')
        print(f"Plotting histograms to: {hist_dir}")
        plot_feature_histograms(features_df, save_dir=hist_dir)

    log_features = [
        "avg_qa_time_minutes", "avg_question_length", "copy_paste_score",
        'full_copy_paste_score', 'answer_seeking_intensity', 'full_answer_seeking_intensity',
        'understanding_signal_strength', 'full_understanding_signal_strength', "qa_turns"
    ]
    for feat in log_features:
        if feat in features_df.columns:
            features_df[feat] = np.log1p(features_df[feat])

    if plot_histograms:
        hist_dir = os.path.join(output_root, 'histograms_after_log')
        print(f"Plotting histograms to: {hist_dir}")
        plot_feature_histograms(features_df, save_dir=hist_dir)

    return features_df
