"""Reviewer Agent (Self-Healing)

Analyses pytest failures and produces precise, actionable fix instructions
for the Coder. Does NOT rewrite code — only diagnoses and prescribes fixes.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ── Node ─────────────────────────────────────────────────────────────────────

def reviewer_node(state: dict) -> dict:
    """LangGraph node: diagnose test failures and write a fix plan."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    test_results = state.get("test_results", {})
    stdout = test_results.get("stdout", "No output captured.")
    stderr = test_results.get("stderr", "")

    # Build code context
    code_context = ""
    for filename, code in state.get("generated_code", {}).items():
        code_context += f"\n# ═══ {filename} ═══\n{code}\n"

    test_context = ""
    for filename, code in state.get("test_code", {}).items():
        test_context += f"\n# ═══ {filename} ═══\n{code}\n"

    fix_attempt = state.get("fix_attempts", 0)

    prompt = f"""You are a senior code reviewer doing a root-cause analysis of test failures.

SOURCE CODE:
{code_context}

TEST CODE:
{test_context}

PYTEST OUTPUT (failures):
{stdout}
{stderr}

Your job: identify every bug causing a test failure and provide a numbered list of EXACT fix instructions.

For each bug:
  1. Which file and function is affected
  2. What the bug is (wrong logic, missing handling, off-by-one, wrong type, etc.)
  3. Exactly how to fix it — be specific (e.g. "change `return x / y` to `if y == 0: raise ValueError; return x / y`")

Do NOT rewrite the code. Just provide the fix instructions.
Be concise — under 400 words total. No preamble."""

    logs = list(state.get("logs", []))
    logs.append(
        f"🔄 [Reviewer] Self-healing loop {fix_attempt + 1}/3 — diagnosing failures…"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        feedback = response.content.strip()

        logs.append(f"📋 [Reviewer] Fix plan ready — routing back to Coder")
        return {"review_feedback": feedback, "logs": logs}

    except Exception as exc:
        logs.append(f"❌ [Reviewer] Error: {exc}")
        return {"review_feedback": f"Review failed: {exc}", "logs": logs}
