"""Security Agent

Two-step security analysis:
  1. Run bandit (via sandbox) on the generated code
  2. Use GPT-4o to interpret the findings, flag HIGH/MEDIUM issues,
     suggest concrete fixes, and assign a plain-English risk summary.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from core.sandbox import run_bandit


# ── Node ─────────────────────────────────────────────────────────────────────

def security_node(state: dict) -> dict:
    """LangGraph node: bandit scan + LLM annotation of security findings."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    run_dir = state.get("run_dir", ".")
    logs = list(state.get("logs", []))
    logs.append("🛡️  [Security] Running bandit security scan…")

    # ── Step 1: bandit ─────────────────────────────────────────────────
    report_json, score = run_bandit(run_dir)

    logs.append(f"📊 [Security] Bandit scan complete — raw score: {score}/10")

    # ── Step 2: LLM annotation ─────────────────────────────────────────
    prompt = f"""You are an application security engineer reviewing a bandit security scan.

BANDIT JSON REPORT:
{report_json[:3000]}

Provide a concise security assessment (max 250 words):

1. **Summary line**: "X HIGH, Y MEDIUM, Z LOW severity issues found."
2. **Critical Issues** (HIGH/MEDIUM only): For each, state:
   - File and line number
   - Issue description (what the vulnerability is)
   - Suggested fix (one sentence)
3. **Overall Rating**: Excellent (9-10) / Good (7-8) / Fair (5-6) / Poor (<5)
4. **Verdict**: One sentence recommendation.

If no HIGH or MEDIUM issues exist, say:
"✅ No significant security issues detected. Code appears safe for demonstration use."

Be direct and technical."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        security_summary = response.content.strip()
    except Exception as exc:
        security_summary = (
            f"Bandit scan complete. Score: {score}/10.\n"
            f"(LLM annotation unavailable: {exc})"
        )

    logs.append(f"✅ [Security] Analysis complete — Security score: {score}/10")

    return {
        "security_report": security_summary,
        "security_score":  score,
        "logs":            logs,
    }
