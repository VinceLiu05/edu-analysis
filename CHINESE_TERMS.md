# Chinese Terms Reference (Modules 1 & 2)

We keep a few Chinese column names/values because the raw dataset is in Chinese. This note clarifies their meanings so research teams can read the code and outputs without ambiguity.

## Column Names
- `学生ID` — student_id
- `教学班ID` — class_id
- `提问时间` — question_time (timestamp of the student's question)
- `提问内容` — question_text (content of the student's question)
- `AI回复` — ai_response (LLM/tutor reply)
- `提问入口` — question_entry/source
  - Common values: `课堂不懂` (confused in class), `课件不懂` (confused about slides/material), `习题不懂` (confused about exercises)
- `起始时间` / `结束时间` — course start/end time
- `开课时间` / `结课时间` — scheduled class start/end time

If you see other Chinese strings in log messages, they come from the original CSV headers/values and are preserved to avoid data loss.

For details on Chinese keyword dictionaries used in feature calculations, see `FEATURE_ENGINEERING.md`.

