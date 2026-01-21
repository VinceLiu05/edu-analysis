# First split by 15-minute time intervals, then use LLM for further segmentation
import pandas as pd
import os
import json
import json5
import re
import time
import random
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from tqdm import tqdm
import argparse
# ==============================    

MAX_WORKERS = 50
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3
# ==============================
# Configuration (can be set via command-line arguments or environment variables)
# ==============================
# API configuration: recommended to set via environment variables, do not hardcode in the code
API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Read from environment variable, or leave empty
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
ENABLE_LLM = os.getenv("ENABLE_LLM", "False").lower() == "true"

# Path configuration: can be set via command-line arguments
INPUT_FOLDER = None  # Will be set in parse_args
OUTPUT_BASE_FOLDER = None  # Will be set in parse_args

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Two-stage dialogue segmentation tool')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Input CSV folder path')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output CSV folder path')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenRouter API key (can also be set via OPENROUTER_API_KEY environment variable)')
    parser.add_argument('--enable_llm', action='store_true',
                       help='Enable LLM segmentation (disabled by default)')
    parser.add_argument('--base_url', type=str, default=None,
                       help='OpenRouter API base URL (can also be set via OPENROUTER_BASE_URL environment variable)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to use (can also be set via OPENROUTER_MODEL environment variable)')
    return parser.parse_args()

# ==============================
# Utility Functions
# ==============================

def robust_read_csv(file_path, text_columns=None):
    """
    Safely read CSV file, suitable for cases containing newlines, Markdown image links, double quotes, etc.
    """
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            engine="python",
            quotechar='"',
            doublequote=True,
            keep_default_na=False,
        )
    except Exception as e:
        print(f"Warning: Failed to read CSV: {file_path}, error: {e}")
        return None
    
    if df.empty:
        print(f"Warning: File {file_path} is empty")
        return None
    
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Check required columns
    # Chinese column names preserved from raw CSVs:
    # - 学生ID (student_id), 提问时间 (question_time)
    required_cols = ["学生ID", "提问时间"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: File {file_path} missing required column: {col}")
            return None
    
    # Process newlines in text columns
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("\n", " ", regex=False)
    
    # Convert time column
    df["提问时间"] = pd.to_datetime(df["提问时间"], errors="coerce")
    df = df.dropna(subset=["提问时间"])
    
    if df.empty:
        print(f"Warning: All rows in file {file_path} are invalid (time parsing failed)")
        return None
    
    return df

# ==============================
# GPT API Calls
# ==============================
def gpt_api_call(messages, model=MODEL_NAME):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return None

# ==============================
# JSON Parsing (Enhanced Robustness)
# ==============================
def robust_json_parse(text):
    """
    Robust JSON parsing that can handle various return formats
    """
    if not text:
        return {}
    
    # Clean text
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    
    try:
        result = json.loads(cleaned)
    except Exception:
        try:
            result = json5.loads(cleaned)
        except Exception:
            print(f"    Warning: JSON parsing failed, raw output: {text[:200]}")
            return {}
    
    # Handle different return formats
    if isinstance(result, dict):
        return result
    elif isinstance(result, list):
        # If list format [1, 1, 2, 2, 3], convert to dictionary
        print(f"    Warning: LLM returned list format, attempting conversion")
        converted = {}
        for idx, session_id in enumerate(result, 1):
            converted[str(idx)] = session_id
        return converted
    else:
        print(f"    Warning: Unknown return format: {type(result)}")
        return {}
        
PROMPT_TEMPLATE = """
You are a dialogue segmentation expert. Analyze the following student-AI tutor interaction logs and segment them into coherent dialogue sessions.

Input: Chronologically ordered interactions with:
- Row number
- Timestamp  
- Student question
- AI response

Segmentation criteria:
1. Preserve chronological order strictly
2. Split when time gap exceeds reasonable conversation pause
3. Split when topic shifts significantly
4. Keep related exchanges in same session

Output: JSON mapping{{row_number: session_id}}, session_id starting from 1
No additional text or explanation.

Interaction logs:
{dialogues_json}
"""
# ==============================
# Stage 0: Question Entry-Based Split (New)
# ==============================
def entrance_based_split(df):
    """
    Split based on changes in question entry
    When the question entry changes, consider it as a different dialogue scenario
    """
    results = []
    if df.empty:
        return results
    
    # Ensure sorted by time
    df = df.sort_values("提问时间").reset_index(drop=True)
    
    current_chunk = [df.iloc[0]]
    current_entrance = df.iloc[0].get("提问入口", "")  # 提问入口 = question entry/source
    
    for i in range(1, len(df)):
        row_entrance = df.iloc[i].get("提问入口", "")
        
        # If question entry changes, split
        if row_entrance != current_entrance:
            results.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
            current_entrance = row_entrance
        else:
            current_chunk.append(df.iloc[i])
    
    # Add the last chunk
    if current_chunk:
        results.append(pd.DataFrame(current_chunk))
    
    print(f"  - Split into {len(results)} segments by question entry")
    for idx, chunk in enumerate(results, 1):
        entrance = chunk.iloc[0].get("提问入口", "未知")
        print(f"    Segment {idx}: entry={entrance}, records={len(chunk)}")
    
    return results

# ==============================
# Stage 1: Time-Based Split (Modified)
# ==============================
def time_based_split(df, time_threshold=15):
    """
    Within the same question entry, split based on time intervals
    """
    results = []
    if df.empty:
        return results
    
    current_chunk = [df.iloc[0]]
    for i in range(1, len(df)):
        delta = (df.iloc[i]["提问时间"] - df.iloc[i - 1]["提问时间"]).total_seconds() / 60
        if delta > time_threshold:
            results.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
        else:
            current_chunk.append(df.iloc[i])
    
    if current_chunk:
        results.append(pd.DataFrame(current_chunk))
    
    return results

# ==============================
# Stage 2: LLM-Based Split
# ==============================
def llm_split(group_df):
    """Use LLM to segment dialogues"""
    try:
        dialogues = []
        
        # Check required columns
        required_columns = ['提问时间', '提问内容', 'AI回复']
        missing_columns = [col for col in required_columns if col not in group_df.columns]
        
        if missing_columns:
            print(f"    Warning: Missing required columns: {missing_columns}, skipping LLM split")
            return {}
        
        # Build dialogue data
        for idx in range(len(group_df)):
            row = group_df.iloc[idx]
            dialogues.append({
                "row_number": idx + 1,
                "timestamp": str(row["提问时间"]),
                "question": str(row["提问内容"])[:500],
                "ai_response": str(row["AI回复"])[:500]
            })
        
        dialogues_json = json.dumps(dialogues, ensure_ascii=False, indent=2)
        
        # Call LLM
        prompt = PROMPT_TEMPLATE.format(dialogues_json=dialogues_json)
        
        messages = [
            {"role": "system", "content": "You are a dialogue segmentation expert. Always output JSON dictionary format, never array."},
            {"role": "user", "content": prompt}
        ]
        
        response = gpt_api_call(messages)
        
        if not response:
            print("    Warning: LLM no response")
            return {}
        
        # Parse result
        parsed_result = robust_json_parse(response)
        
        # Verify again if it's a dictionary
        if not isinstance(parsed_result, dict):
            print(f"    Warning: Still not dictionary format after parsing: {type(parsed_result)}")
            return {}
        
        # Ensure all keys are strings and values are integers
        result = {}
        for key, value in parsed_result.items():
            try:
                result[str(key)] = int(value)
            except (ValueError, TypeError) as e:
                print(f"    Warning: Failed to convert key-value pair: {key}={value}, error: {e}")
                continue
        
        return result
        
    except Exception as e:
        print(f"    Error: LLM split failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# ==============================
# Process Single Student Data (Enhanced Error Handling)
# ==============================
def process_student_data(student_df, student_id, output_folder):
    """Process all dialogue data for a single student (supports time+LLM dual segmentation, and safe saving)"""
    # Sort by time
    student_df = student_df.sort_values("提问时间").reset_index(drop=True)
    
    print(f"  Starting to process student {student_id} data...")
    print(f"  - Total records: {len(student_df)}")
    
    # Stage 0: First split by question entry
    if "提问入口" in student_df.columns:
        entrance_splits = entrance_based_split(student_df)
    else:
        print("  - No '提问入口' column, skipping entry-based split")
        entrance_splits = [student_df]
    
    file_index = 1  # Counter for each student's own numbering

    # Iterate through each "question entry" segment
    for entrance_idx, entrance_group in enumerate(entrance_splits, start=1):
        entrance_name = entrance_group.iloc[0].get("提问入口", "未知") if "提问入口" in entrance_group.columns else "未知"
        print(f"\n  Processing entry segment {entrance_idx}/{len(entrance_splits)}: {entrance_name}")
        
        # Stage 1: Time-based split
        time_splits = time_based_split(entrance_group)
        print(f"    - Time split into {len(time_splits)} sub-segments")

        # Stage 2: LLM split (optional)
        for _, group in enumerate(time_splits, start=1):
            group = group.reset_index(drop=True)
            
            # If segment is too small, save directly
            if len(group) <= 2:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}_{timestamp}.csv")
                group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    Small segment, saving directly: {os.path.basename(output_csv)} ({len(group)} rows)")
                file_index += 1
                continue

            # If LLM not enabled, save directly
            if not ENABLE_LLM:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}_{timestamp}.csv")
                group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    Skipping LLM split, saving directly: {os.path.basename(output_csv)} ({len(group)} rows)")
                file_index += 1
                continue

            # Enable LLM split
            mapping = llm_split(group)
            if not mapping:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}_{timestamp}.csv")
                group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    Warning: LLM invalid or failed, saving original segment: {os.path.basename(output_csv)} ({len(group)} rows)")
                file_index += 1
                continue

            # Save sub-segments based on LLM mapping
            group["session_id"] = group.index.map(lambda i: mapping.get(str(i+1), None))
            for session_id, sub_df in group.groupby("session_id"):
                if sub_df.empty:
                    continue
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}_{timestamp}.csv")
                sub_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"    Saved LLM sub-segment: {os.path.basename(output_csv)} ({len(sub_df)} rows, session={session_id})")
                file_index += 1

    return file_index - 1

# ==============================
# Process Single CSV File
# ==============================
def process_csv_file(file_path):
    """Process a single CSV file, group by student ID and process separately"""
    try:
        # Get filename (without extension)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create corresponding output folder
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, base_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting to process file: {base_name}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*60}")
        
        # Read CSV
        df = robust_read_csv(file_path, text_columns=["提问内容", "AI回复"])
        if df is None or df.empty:
            print(f"Warning: {file_path} has no valid data, skipping")
            return
        
        # Group by student ID
        student_groups = df.groupby("学生ID")
        total_students = len(student_groups)
        
        print(f"Found {total_students} students' dialogue records")
        total_dialogues = 0
        
        # Use tqdm to show student progress
        for student_idx, (student_id, student_df) in enumerate(
            tqdm(student_groups, total=total_students, desc=f"Processing students({base_name})", ncols=80)
        ):
            # student_groups is (id, group) pairs, need to manually unpack
            print(f"\n[{student_idx+1}/{total_students}] Processing student {student_id} data...")
            print(f"  - Total records: {len(student_df)}")
            
            # Process all dialogues for this student
            dialogue_count = process_student_data(student_df, student_id, output_folder)
            total_dialogues += dialogue_count
            
            print(f"  - Generated dialogues: {dialogue_count}")
        
        print(f"\nFile {base_name} processing complete!")
        print(f"  - Number of students: {total_students}")
        print(f"  - Total dialogues: {total_dialogues}")
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to process file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================
# Main Program
# ==============================
def main():
    global ENABLE_LLM, INPUT_FOLDER, OUTPUT_BASE_FOLDER, API_KEY, BASE_URL, MODEL_NAME

    args = parse_args()
    
    # Set paths
    INPUT_FOLDER = args.input_folder
    OUTPUT_BASE_FOLDER = args.output_folder
    os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
    
    # Set API configuration (priority: command-line arguments, then environment variables, finally default values)
    if args.api_key:
        API_KEY = args.api_key
    elif not API_KEY:
        API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    if args.base_url:
        BASE_URL = args.base_url
    else:
        BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if args.model:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    
    ENABLE_LLM = args.enable_llm

    print(f"LLM segmentation enabled: {'Yes' if ENABLE_LLM else 'No'}\n")
    if ENABLE_LLM and not API_KEY:
        print("Warning: LLM enabled but API key not set, LLM functionality will be unavailable")
        print("   Please set API key via --api_key parameter or OPENROUTER_API_KEY environment variable\n")

    # Get all CSV files
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        print("Warning: No CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output base folder: {OUTPUT_BASE_FOLDER}")
    print(f"{'='*60}\n")

    success_count = 0
    failed_files = []

    # Use tqdm to wrap file processing progress
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_csv_file, os.path.join(INPUT_FOLDER, f)): f 
            for f in csv_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="File processing progress", ncols=80):
            file_name = futures[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_files.append(file_name)
            except Exception as e:
                print(f"Error: Failed to process {file_name}: {e}")
                failed_files.append(file_name)

    print(f"\n{'='*60}")
    print(f"All files processed!")
    print(f"Success: {success_count}/{len(csv_files)}")

    if failed_files:
        print(f"Failed files:")
        for f in failed_files:
            print(f"  - {f}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()