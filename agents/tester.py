"""Tester Agent

Given the generated source code, writes a comprehensive pytest test suite
covering happy paths, edge cases, and exception handling.
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

def tester_node(state: dict) -> dict:
    """LangGraph node: generate pytest unit tests for every source file."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    # Build source code context
    code_context = ""
    for filename, code in state.get("generated_code", {}).items():
        code_context += f"\n# ═══ {filename} ═══\n{code}\n"

    prompt = f"""You are a senior QA engineer writing pytest unit tests.

ORIGINAL REQUIREMENT:
{state['requirement']}

SOURCE CODE TO TEST:
{code_context}

Return a JSON object where:
  • Each KEY   is a test filename (must start with "test_", e.g. "test_calculator.py")
  • Each VALUE is the COMPLETE pytest code for that test file

Example:
{{
  "test_calculator.py": "import pytest\\nfrom calculator import add\\n\\ndef test_add_positive():\\n    assert add(2, 3) == 5\\n\\ndef test_add_negative():\\n    assert add(-1, -2) == -3\\n"
}}

Testing rules:
  • Import source modules directly by name (e.g. `from calculator import Calculator`)
  • Cover EVERY public function and method
  • Include: happy-path tests, edge cases (zero, empty, None), and exception tests using pytest.raises()
  • Use plain pytest functions (not unittest.TestCase)
  • Keep fixtures minimal; prefer direct function calls
  • Return ONLY the raw JSON — no markdown fences, no extra text

JSON:"""

    logs = list(state.get("logs", []))
    logs.append("🧪 [Tester] Writing unit tests…")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        test_dict = _extract_json(response.content)

        if not isinstance(test_dict, dict):
            raise ValueError("LLM returned a non-dict response")

        logs.append(f"✅ [Tester] Tests ready: {list(test_dict.keys())}")
        return {"test_code": test_dict, "logs": logs}

    except Exception as exc:
        logs.append(f"❌ [Tester] Error: {exc}")
        return {
            "test_code": {"test_main.py": f"# Test generation failed: {exc}\n"},
            "logs": logs,
        }
