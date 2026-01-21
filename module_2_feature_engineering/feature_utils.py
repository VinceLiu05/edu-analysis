"""Feature utility functions for QA dialog analysis."""

from typing import Tuple
import re
import numpy as np
import pandas as pd

# COPY_KEYWORDS: Chinese cues that likely come from pasted homework/exam text
COPY_KEYWORDS = [
    "如下", "如上", "这道题", "怎么做", "哪里错了",
    "A.", "B.", "C.", "D.",
    "做一下", "怎么写", "选什么", "解一下", "咋做", "答案", "结果", "求解",
    "解答", "回答", "解题", "单选题", "多选题",
    "描述错误的是", "描述正确的是", "回答下列问题", "a.", "b.", "c.", "d.",
    "有什么作用", "请上传你的文件", "please upload the file"
]

# Numeric patterns seen in Chinese exam text
COPY_KEYWORDS_PATTERNS = [
    r'\d+分',
    r'第\d+题'
]

# Phrases indicating direct answer seeking in prompts
ANSWER_SEEKING_KEYWORDS_PROMPT = [
    "答案", "选什么", "选哪个", "做一下", "解一下", "求解", "答案是什么",
    "帮我做", "直接告诉我", "怎么选", "结果是", "最终结果", "帮我算",
    "正确答案", "标准答案", "给我答案"
]

# Phrases indicating the AI reply is discussing exam answers/options
ANSWER_SEEKING_KEYWORDS_AI = [
    "题目", "练习题", "习题", "选择题", "多选题", "单选题", "填空题",
    "这道题", "这题", "答案", "选项", "正确答案", "应该选"
]

# Phrases indicating conceptual understanding intent
UNDERSTANDING_KEYWORDS_PROMPT = [
    "为什么", "为啥", "怎么会", "原因", "机制", "如何理解", "怎么理解",
    "我的理解是", "我觉得", "是不是", "是因为", "区别", "不同",
    "相比", "比较", "有什么联系", "关系",
    "我尝试", "我试过", "关键在于", "核心是", "本质",
    "如果", "那如果", "会怎样", "会不会", 
]

UNDERSTANDING_KEYWORDS_AI = [
    "让我们", "我们可以", "首先", "然后", "因为", "所以",
    "理解", "概念", "原理", "思路", "关键", "本质",
    "举个例子", "比如", "类似", "可以这样想", "换句话说"
]

NON_NATURAL_PATTERNS = [
    r'\d+[\.、]\s*[A-D选项]',
    r'[（\(]\s*\d+\s*分\s*[）\)]',
    r'```',
    r'_{3,}',
    r'\[\s*\]',
    r'[A-D][\.、)]\s*[^\n]{1,50}\s*[A-D][\.、)]',
]

FEATURE_COLUMNS = [
    'qa_turns', 'avg_qa_time_minutes', 'avg_question_length',
    'copy_paste_score', 'answer_seeking_intensity', 'understanding_signal_strength',
    'course_progress_ratio', 'is_exam_week', 'day_period', 'is_in_class_time'
]

def debug_infinite_values(features_df: pd.DataFrame) -> None:
    """Debug and print infinite and NaN values in features DataFrame."""
    print("\n" + "=" * 50)
    print("Checking Infinite Values")
    print("=" * 50)

    print("\nAvailable columns in features_df:")
    print(features_df.columns.tolist())

    has_issues = False

    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            continue

        inf_count = np.isinf(features_df[col]).sum()
        nan_count = np.isnan(features_df[col]).sum()

        if inf_count > 0 or nan_count > 0:
            has_issues = True
            print(f"\nColumn '{col}':")
            print(f"   - Infinite values: {inf_count}")
            print(f"   - NaN values: {nan_count}")

            if inf_count > 0:
                inf_indices = features_df[np.isinf(features_df[col])].index
                print(f"   - Infinite values in rows: {inf_indices.tolist()[:5]}...")

                if 'file_name' in features_df.columns:
                    sample_files = features_df.loc[inf_indices[:3], 'file_name'].tolist()
                    print(f"   - Sample files: {sample_files}")

        finite_data = features_df[col][np.isfinite(features_df[col])]
        if not finite_data.empty:
            print(f"\n{col} (finite values only):")
            print(f"   - Min:  {finite_data.min():.2f}")
            print(f"   - Max:  {finite_data.max():.2f}")
            print(f"   - Mean: {finite_data.mean():.2f}")
            print(f"   - Std:  {finite_data.std():.2f}")
        else:
            print(f"\n{col}: All values are infinite or NaN")

    if not has_issues:
        print("\nNo infinite or NaN values found in feature columns")

    print("=" * 50)


def normalize_for_keyword(text: str) -> str:
    """Normalize text for keyword detection (full-width to half-width, remove whitespace)."""
    if text is None:
        return ""

    s = str(text)

    result = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    s = "".join(result)

    s = re.sub(r'\s+', '', s)

    return s


def detect_has_image(prompts: pd.Series) -> int:
    """Detect if prompts contain images. Returns 1 if yes, 0 otherwise."""
    image_pattern = r'!\[.*?\]\(.*?\)|<img\b'
    has_image = prompts.astype(str).str.contains(image_pattern, regex=True).any()
    return int(has_image)

def calculate_copy_paste_score(
    prompts: pd.Series,
    is_confusion_entry: int = 0
) -> Tuple[int, int]:
    """Detect copy-paste signals. Returns (is_copy_paste, copy_paste_count)."""
    count = 0
    is_copy_paste = 0

    has_image = detect_has_image(prompts)
    if has_image and is_confusion_entry == 0:
        count += 1
        is_copy_paste = 1

    avg_length = prompts.astype(str).str.len().mean()
    if avg_length > 500:
        count += 1

    all_text = " ".join(prompts.astype(str))
    for pattern in NON_NATURAL_PATTERNS:
        matches = re.findall(pattern, all_text)
        count += len(matches)

    normalized = normalize_for_keyword(all_text)
    for kw in COPY_KEYWORDS:
        count += normalized.count(kw)

    if count > 0:
        is_copy_paste = 1

    return is_copy_paste, int(count)

def calculate_answer_seeking_intensity(
    prompts: pd.Series,
    ai_responses: pd.Series,
    has_image: int,
    is_confusion_entry: int = 0
) -> int:
    """Calculate answer-seeking pattern occurrences."""
    total_count = 0
    all_prompts = " ".join(prompts.astype(str))
    normalized_prompts = normalize_for_keyword(all_prompts)

    for kw in ANSWER_SEEKING_KEYWORDS_PROMPT:
        total_count += normalized_prompts.count(kw)

    if len(ai_responses) > 0 and pd.notna(ai_responses.iloc[0]):
        first_ai_reply = str(ai_responses.iloc[0])[:100]
        normalized_ai = normalize_for_keyword(first_ai_reply)
        for kw in ANSWER_SEEKING_KEYWORDS_AI:
            total_count += normalized_ai.count(kw)

    if has_image == 1 and is_confusion_entry == 0:
        total_count += 1

    return int(total_count)

def calculate_understanding_signal_strength(
    prompts: pd.Series
) -> int:
    """Calculate understanding-related keyword occurrences."""
    total_count = 0

    all_prompts = " ".join(prompts.astype(str))
    normalized_prompts = normalize_for_keyword(all_prompts)

    for kw in UNDERSTANDING_KEYWORDS_PROMPT:
        total_count += normalized_prompts.count(kw)

    hypothetical_patterns = [r'如果.*会', r'那如果', r'可以.*吗']
    for pattern in hypothetical_patterns:
        matches = re.findall(pattern, all_prompts)
        total_count += len(matches)

    return int(total_count)