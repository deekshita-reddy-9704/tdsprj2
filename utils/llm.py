import os
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from openai import OpenAI

# ---- System prompts (strict JSON) ----
SYSTEM_PLANNER = """You are a planning assistant that outputs STRICT JSON (no prose).
Inputs:
- 'original_text': the entire questions.txt text.
- 'expected_format': 'array' or 'object' (inferred by server).
- 'desired_array_length': integer (usually 4) if expected_format == 'array'.
- 'questions_list': a list of individual questions extracted from questions.txt.
- 'files': uploaded files metadata (name, suffix, etc.).
- 'tables': list of registered DuckDB table names.

Output JSON object:
{
  "format": "array" | "object",
  "array_length": number | null,
  "tasks": [
    // choose among:
    // {"type":"scrape_wikipedia","page":"<Title or slug>","table_index":0,"columns":null|[...],"top_k":50,"key":"alias"}
    // {"type":"duckdb_sql","sql":"SELECT ...","key":"alias"}
    // {"type":"plot_scatter","table":"<table>","x":"<col>","y":"<col>","hue":null|"<col>","label_x":"Rank","label_y":"Peak","key":"alias"}
    // {"type":"noop"}
  ],
  "response_schema_hint": {} | null
}

Rules:
- Keep tasks minimal and fast (finish < 3 minutes).
- Only plan for Wikipedia scraping if useful (e.g., highest-grossing films).
- If tables exist, plan SQL queries accordingly.
- For array format, ALWAYS set array_length to the provided desired_array_length (default 4).
- For object format, you MUST assume the final object keys will exactly match 'questions_list' (do NOT rephrase keys).
- DO NOT include text outside JSON.
"""

SYSTEM_COMPOSER = """You are a composition assistant. Produce ONLY valid JSON per the plan and strict format rules.

You will receive:
- original_text: entire questions.txt
- expected_format: 'array' or 'object'
- desired_array_length: when array, usually 4 (must be exact)
- questions_list: exact ordered list of questions (keys must match if object)
- plan: planning JSON (already validated)
- artifacts: results of tool tasks (tables/lists/plots data URI)
- errors: execution errors

Your job:
- If 'expected_format' == 'array', return a JSON array of EXACTLY 'desired_array_length' elements.
- If 'expected_format' == 'object', return a JSON object whose keys are EXACTLY 'questions_list' (string match).
- Use the 'artifacts' to answer when possible. For plots, return a data URI under 100k bytes under a reasonable element/key.
- If info is missing or an error occurred, return null for that element/key. Do not write explanations.
- Output ONLY JSON. No markdown, no commentary.
"""

class PlanResult(BaseModel):
    format: str
    array_length: Optional[int] = None
    tasks: List[Dict[str, Any]]
    response_schema_hint: Optional[Dict[str, Any]] = None


class LLMPlanner:
    """OpenAI wrapper for plan & compose with strict JSON outputs and schema hints."""
    def __init__(self, model_planner: str = None, model_composer: str = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")
        self.client = OpenAI(api_key=api_key)
        self.model_planner = model_planner or "gpt-4o-mini"
        self.model_composer = model_composer or "gpt-4o-mini"

    async def plan(
        self,
        questions_text: str,
        files_meta: List[Dict[str, Any]],
        tables: Dict[str, str],
        expected_format: str,
        desired_array_length: int,
        questions_list: List[str],
    ) -> PlanResult:
        messages = [
            {"role": "system", "content": SYSTEM_PLANNER},
            {"role": "user", "content": json.dumps({
                "original_text": questions_text,
                "expected_format": expected_format,
                "desired_array_length": desired_array_length,
                "questions_list": questions_list,
                "files": files_meta,
                "tables": list(tables.keys()),
            })}
        ]
        resp = self.client.chat.completions.create(
            model=self.model_planner,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        obj = json.loads(content)
        return PlanResult(**obj)

    async def compose_answer(
        self,
        original_text: str,
        expected_format: str,
        desired_array_length: int,
        questions_list: List[str],
        plan: PlanResult,
        artifacts: Dict[str, Any],
        errors: List[str],
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_COMPOSER},
            {"role": "user", "content": json.dumps({
                "original_text": original_text,
                "expected_format": expected_format,
                "desired_array_length": desired_array_length,
                "questions_list": questions_list,
                "plan": plan.model_dump(),
                "artifacts": artifacts,
                "errors": errors
            })}
        ]

        # Use response_format JSON schema for arrays to force exact length
        if expected_format == "array":
            resp = self.client.chat.completions.create(
                model=self.model_composer,
                messages=messages,
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fixed_array",
                        "schema": {
                            "type": "array",
                            "minItems": desired_array_length,
                            "maxItems": desired_array_length,
                            "items": {}
                        },
                        "strict": False
                    }
                },
            )
        else:
            resp = self.client.chat.completions.create(
                model=self.model_composer,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

        return resp.choices[0].message.content
