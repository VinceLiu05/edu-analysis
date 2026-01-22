# Chinese Terms Reference (Modules 1 & 2)

We keep a few Chinese column names/values because the raw dataset is in Chinese. This note clarifies their meanings so research teams can read the code and outputs without ambiguity.

## High-frequency columns
- `学生ID` — student_id
- `教学班ID` — class_id
- `提问时间` — question_time (timestamp of the student’s question)
- `提问内容` — question_text (content of the student’s question)
- `AI回复` — ai_response (LLM/tutor reply)
- `提问入口` — question_entry/source
  - Common values: `课堂不懂` (confused in class), `课件不懂` (confused about slides/material), `习题不懂` (confused about exercises)
- `起始时间` / `结束时间` — course start/end time
- `开课时间` / `结课时间` — scheduled class start/end time

## Lower-frequency phrases inside feature dictionaries
- Copy-paste cues (`COPY_KEYWORDS`): phrases like “这道题”“怎么做”“答案” that suggest homework/exam text was pasted.
- Answer-seeking cues (`ANSWER_SEEKING_KEYWORDS_*`): phrases like “正确答案”“选什么” indicating the student is directly asking for answers.
- Understanding cues (`UNDERSTANDING_KEYWORDS_PROMPT`): phrases like “为什么”“如何理解” indicating conceptual understanding intent.

If you see other Chinese strings in log messages, they come from the original CSV headers/values and are preserved to avoid data loss.

