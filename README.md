# Data Analyst Agent

A production-ready **FastAPI** service that accepts `questions.txt` (+ optional files), uses an **LLM** to plan actions, and performs:
- Web scraping (Wikipedia)
- DuckDB SQL on uploaded CSV/Parquet/JSON
- Pandas cleaning/analysis
- Plotting (Matplotlib/Seaborn) with strict **grader compliance**:
  - scatter of X vs Y
  - **dotted red** regression line
  - labelled axes
  - PNG **< 100 KB**, returned as **data URI**

The API **always** returns the correct JSON shape:
- If a **JSON array** is requested → **exactly 4 elements** (pad/truncate).
- If a **JSON object** is requested → keys **exactly match** the questions parsed from `questions.txt`.

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here   # PowerShell: $env:OPENAI_API_KEY="your_key_here"
uvicorn main:app --reload
