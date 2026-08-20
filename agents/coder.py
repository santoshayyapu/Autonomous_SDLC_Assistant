"""Coder Agent

Given the technical spec (+ optional reviewer feedback from a prior failed
attempt), generates complete, working Python source code for every planned file.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) >= 3 else parts[-1]
        if text.startswith("json\n"):
            text = text[5:]
    return json.loads(text.strip())


# ── Node ─────────────────────────────────────────────────────────────────────

def coder_node(state: dict) -> dict:
    """LangGraph node: generate (or regenerate) Python code for each file."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    fix_context = ""
    fix_attempts = state.get("fix_attempts", 0)

    review_feedback = state.get("review_feedback", "")
    if review_feedback:
        fix_context = f"""
⚠️  SELF-HEALING LOOP — Attempt {fix_attempts + 1}/3
The previous code version failed its tests. You MUST address every issue below:

{review_feedback}

Do not repeat the same mistakes.
"""

    file_list = "\n".join(f"  - {f}" for f in state.get("file_structure", ["main.py"]))
    tone = state.get("tone", "Standard")

    prompt = f"""You are a senior Python developer writing production-quality code.

REQUIREMENT:
{state['requirement']}

CODING TONE/STYLE:
{tone}

TECHNICAL SPEC:
{state.get('technical_spec', state['requirement'])}

FILES TO CREATE:
{file_list}
{fix_context}
Return a JSON object where:
  • Each KEY   is a filename  (e.g. "calculator.py")
  • Each VALUE is the COMPLETE Python source code for that file

Example:
{{
  "calculator.py": "def add(a, b):\\n    return a + b\\n"
}}

Coding rules:
  • Python standard library ONLY — no third-party imports
  • Complete code — NO placeholders, no "pass", no TODO stubs
  • Robust error handling (raise ValueError / TypeError where appropriate)
  • PEP 8 compliant (snake_case, 4-space indent)
  • No test logic in source files
  • Return ONLY the raw JSON object — no markdown fences, no extra text

JSON:"""

    logs = list(state.get("logs", []))
    action = "🔄 Regenerating" if review_feedback else "💻 Generating"
    logs.append(f"{action} [Coder] Writing Python code for {state.get('file_structure', [])}")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        code_dict = _extract_json(response.content)

        if not isinstance(code_dict, dict):
            raise ValueError("LLM returned a non-dict response")

        new_attempts = fix_attempts + (1 if review_feedback else 0)
        logs.append(f"✅ [Coder] Code ready: {list(code_dict.keys())}")

        return {
            "generated_code": code_dict,
            "fix_attempts":   new_attempts,
            "review_feedback": "",          # Clear feedback after applying it
            "logs": logs,
        }

    except Exception as exc:
        logs.append(f"❌ [Coder] Error: {exc}")
        return {
            "generated_code": {"main.py": f"# Code generation failed: {exc}\n"},
            "fix_attempts":   fix_attempts,
            "logs": logs,
        }
