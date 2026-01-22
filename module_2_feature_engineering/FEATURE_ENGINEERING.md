# Feature Engineering Documentation

This document describes the 10 features extracted from segmented dialog sessions in Module 2, along with their calculation formulas and the Chinese keyword dictionaries used.

## Feature Overview

The 10 features are categorized into three groups:

- **Behavioral engagement** (3 features):
  - Number of Conversation Turns (`qa_turns`)
  - Average Time Duration per Turn (`avg_qa_time_minutes`)
  - Average Word Count per Turn (`avg_question_length`)
- **Cognitive engagement** (3 features):
  - Number of Copy-Paste Events (`copy_paste_score`)
  - Number of Direct Answer Requests (`answer_seeking_intensity`)
  - Number of Understanding-Oriented Queries (`understanding_signal_strength`)
- **Temporal engagement** (4 features):
  - Week Progress (`course_progress_ratio`)
  - Exam Period Indicator (`is_exam_week`)
  - Time of Day (`day_period`)
  - In-Class Indicator (`is_in_class_time`)

## Feature Name Mapping (Paper vs. Code)

The following table maps the feature names used in the paper to the names used in the code:

| Paper Name | Code Name | Dimension |
|------------|-----------|-----------|
| Number of Conversation Turns | `qa_turns` | Behavioral Engagement |
| Average Time Duration per Turn | `avg_qa_time_minutes` | Behavioral Engagement |
| Average Word Count per Turn | `avg_question_length` | Behavioral Engagement |
| Number of Copy-Paste Events | `copy_paste_score` | Cognitive Engagement |
| Number of Direct Answer Requests | `answer_seeking_intensity` | Cognitive Engagement |
| Number of Understanding-Oriented Queries | `understanding_signal_strength` | Cognitive Engagement |
| Week Progress | `course_progress_ratio` | Temporal Engagement |
| Exam Period Indicator | `is_exam_week` | Temporal Engagement |
| Time of Day | `day_period` | Temporal Engagement |
| In-Class Indicator | `is_in_class_time` | Temporal Engagement |

## Feature Definitions and Formulas

### Behavioral Engagement Features

#### 1. `qa_turns`
- **Definition**: Number of question-answer turns in the dialog session
- **Formula**: `qa_turns = len(df)` (count of rows in the dialog CSV)
- **Type**: Integer

#### 2. `avg_qa_time_minutes`
- **Definition**: Average time interval (in minutes) between consecutive Q&A turns
- **Formula**: 
  - If `qa_turns > 1`: `total_time = max(0, (max(提问时间) - min(提问时间)).total_seconds() / 60)`
  - Then: `avg_qa_time = total_time / (qa_turns - 1)` if `qa_turns > 1` and `total_time > 0`, else `0`
  - **Special handling**: For single-turn dialogs (`qa_turns == 1`), the value is adjusted using the median of 2-turn dialogs from the same class (or globally) divided by 3
- **Type**: Float (minutes)

#### 3. `avg_question_length`
- **Definition**: Average length of question text after removing markdown images, HTML img tags, and certain boilerplate text
- **Formula**: 
  - For each question: remove `![...](...)`, `<img>`, "请上传你的文件（Please upload the file）", and "Notice/注意" lines
  - Then: `avg_question_length = mean(question_lengths)` across all questions in the session
- **Type**: Float (characters)

### Cognitive Engagement Features

#### 4. `copy_paste_score`
- **Definition**: Count of signals indicating that homework/exam text was likely copy-pasted into the question
- **Formula**: 
  ```
  count = 0
  if has_image and is_confusion_entry == 0:
      count += 1
  if avg_question_length > 500:
      count += 1
  count += matches(NON_NATURAL_PATTERNS in all_text)
  count += sum(occurrences(COPY_KEYWORDS in normalized_text))
  copy_paste_score = count
  ```
- **Type**: Integer (count)

#### 5. `answer_seeking_intensity`
- **Definition**: Count of signals indicating the student is directly seeking answers rather than understanding
- **Formula**: 
  ```
  total_count = 0
  total_count += sum(occurrences(ANSWER_SEEKING_KEYWORDS_PROMPT in normalized_prompts))
  if first_ai_reply exists:
      total_count += sum(occurrences(ANSWER_SEEKING_KEYWORDS_AI in normalized_first_ai_reply))
  if has_image == 1 and is_confusion_entry == 0:
      total_count += 1
  answer_seeking_intensity = total_count
  ```
- **Type**: Integer (count)

#### 6. `understanding_signal_strength`
- **Definition**: Count of signals indicating the student is seeking conceptual understanding
- **Formula**: 
  ```
  total_count = 0
  total_count += sum(occurrences(UNDERSTANDING_KEYWORDS_PROMPT in normalized_prompts))
  total_count += matches(hypothetical_patterns: r'如果.*会', r'那如果', r'可以.*吗')
  understanding_signal_strength = total_count
  ```
- **Type**: Integer (count)

### Temporal Engagement Features

#### 7. `course_progress_ratio`
- **Definition**: Week number (1-20) when the dialog occurred, relative to the course start
- **Formula**: 
  - `days_total = (course_end - course_start).days`
  - `days_per_week = days_total / 20`
  - `week_progress = floor((qa_start_time - course_start).days / days_per_week) + 1`
  - Clamped to range [1, 20]
- **Type**: Integer (week number, 1-20)

#### 8. `is_exam_week`
- **Definition**: Binary indicator (0 or 1) if the dialog occurred within 28 days before course end
- **Formula**: 
  - `window_start = course_end - 28 days`
  - `is_exam_week = 1` if `window_start <= qa_start_time <= course_end`, else `0`
- **Type**: Binary (0 or 1)

#### 9. `day_period`
- **Definition**: Hour period of the day (1-24) when the dialog started, with 8:00 AM as period 1
- **Formula**: 
  - `raw_day_period = hour + minute / 60.0`
  - `shifted = (raw_day_period - 8.0) % 24.0`
  - `day_period = floor(shifted) + 1`
- **Type**: Integer (1-24)

#### 10. `is_in_class_time`
- **Definition**: Binary indicator (0 or 1) if any question in the dialog occurred during scheduled class time
- **Formula**: 
  - For each question timestamp, check if it falls within any scheduled class period (`开课时间` to `结课时间`) for the class
  - `is_in_class_time = 1` if any question is in class time, else `0`
- **Type**: Binary (0 or 1)

## Keyword Dictionaries

The following Chinese keyword dictionaries are used in cognitive engagement feature calculations. These phrases are preserved from the original dataset and are normalized (full-width to half-width, whitespace removed) before matching.

### Copy-Paste Cues (`COPY_KEYWORDS`)
Phrases that suggest homework/exam text was pasted:
- "如下", "如上", "这道题", "怎么做", "哪里错了"
- "A.", "B.", "C.", "D."
- "做一下", "怎么写", "选什么", "解一下", "咋做", "答案", "结果", "求解"
- "解答", "回答", "解题", "单选题", "多选题"
- "描述错误的是", "描述正确的是", "回答下列问题", "a.", "b.", "c.", "d."
- "有什么作用", "请上传你的文件", "please upload the file"

### Copy-Paste Patterns (`COPY_KEYWORDS_PATTERNS`)
Regex patterns for numeric exam text:
- `r'\d+分'` (e.g., "5分", "10分")
- `r'第\d+题'` (e.g., "第1题", "第10题")

### Non-Natural Patterns (`NON_NATURAL_PATTERNS`)
Regex patterns indicating structured exam/homework text:
- `r'\d+[\.、]\s*[A-D选项]'` (numbered options)
- `r'[（\(]\s*\d+\s*分\s*[）\)]'` (points in parentheses)
- `r'```'` (code blocks)
- `r'_{3,}'` (multiple underscores)
- `r'\[\s*\]'` (empty brackets)
- `r'[A-D][\.、)]\s*[^\n]{1,50}\s*[A-D][\.、)]'` (multiple options in sequence)

### Answer-Seeking Cues

#### `ANSWER_SEEKING_KEYWORDS_PROMPT`
Phrases in student prompts indicating direct answer seeking:
- "答案", "选什么", "选哪个", "做一下", "解一下", "求解", "答案是什么"
- "帮我做", "直接告诉我", "怎么选", "结果是", "最终结果", "帮我算"
- "正确答案", "标准答案", "给我答案"

#### `ANSWER_SEEKING_KEYWORDS_AI`
Phrases in AI responses indicating discussion of exam answers/options:
- "题目", "练习题", "习题", "选择题", "多选题", "单选题", "填空题"
- "这道题", "这题", "答案", "选项", "正确答案", "应该选"

### Understanding Cues

#### `UNDERSTANDING_KEYWORDS_PROMPT`
Phrases in student prompts indicating conceptual understanding intent:
- "为什么", "为啥", "怎么会", "原因", "机制", "如何理解", "怎么理解"
- "我的理解是", "我觉得", "是不是", "是因为", "区别", "不同"
- "相比", "比较", "有什么联系", "关系"
- "我尝试", "我试过", "关键在于", "核心是", "本质"
- "如果", "那如果", "会怎样", "会不会"

#### `UNDERSTANDING_KEYWORDS_AI`
Phrases in AI responses indicating explanatory/teaching content:
- "让我们", "我们可以", "首先", "然后", "因为", "所以"
- "理解", "概念", "原理", "思路", "关键", "本质"
- "举个例子", "比如", "类似", "可以这样想", "换句话说"

### Hypothetical Patterns
Regex patterns for hypothetical questions (used in `understanding_signal_strength`):
- `r'如果.*会'` (if...then)
- `r'那如果'` (what if)
- `r'可以.*吗'` (can...?)

## Feature Post-Processing

After extraction, the following features are log-transformed (using `log1p`) before clustering:
- `avg_qa_time_minutes`
- `avg_question_length`
- `copy_paste_score`
- `answer_seeking_intensity`
- `understanding_signal_strength`
- `qa_turns`

This transformation helps normalize the distributions and reduce the impact of outliers.

