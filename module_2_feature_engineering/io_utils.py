import pandas as pd
import os

def load_reference_data(class_time_file, class_schedule_file,
                        school_info_file=None, class_info_file=None):
    """Load reference data (class time, schedule, school info, class info)."""
    print("Loading reference CSV files...")

    df_class = pd.read_csv(class_time_file, encoding='utf-8-sig')
    df_schedule = pd.read_csv(class_schedule_file, encoding='utf-8-sig')

    for col in ['开始时间', '结束时间']:
        if col in df_class.columns:
            df_class[col] = pd.to_datetime(df_class[col], errors='coerce')
    for col in ['开课时间', '结课时间']:
        if col in df_schedule.columns:
            df_schedule[col] = pd.to_datetime(df_schedule[col], errors='coerce')

    df_school = pd.DataFrame()
    if school_info_file and pd.io.common.file_exists(school_info_file):
        try:
            df_school = pd.read_csv(school_info_file, encoding='utf-8-sig')
            for col in ['起始时间', '结束时间']:
                if col in df_school.columns:
                    df_school[col] = pd.to_datetime(df_school[col], errors='coerce')
            print(f"Loaded school info: {len(df_school)} rows from {school_info_file}")
        except Exception as e:
            print(f"Failed to load school info file: {e}")

    df_class_info = pd.DataFrame()
    if class_info_file and os.path.exists(class_info_file):
        df_class_info = pd.read_csv(class_info_file, encoding='utf-8-sig')
        print(f"Loaded class info: {len(df_class_info)} rows")

    df_school_bundle = {
        'df_school': df_school,
        'df_class_info': df_class_info
    }

    return df_class, df_schedule, df_school_bundle
