"""Baseline Pipeline — Single-LLM Approach

One prompt to GPT-4o that requests both source code and tests in a single
response.  No architecture planning, no self-healing loop.

Used as the comparison baseline in the Research benchmark.
"""

import os
import time
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.sandbox import prepare_run_dir, run_pytest, run_pylint, run_bandit
from integrations.github import push_to_github


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


# ── Public API ────────────────────────────────────────────────────────────────

def run_baseline_pipeline(requirement: str, tone: str, state_github: dict, run_id: str, run_dir: str) -> dict:
    """
    Single-LLM baseline: one prompt → code + tests → sandbox → PR.

    Parameters
    ----------
    requirement : str   Natural-language software requirement.
    tone        : str   Tone of coding.
    run_id      : str   Short UUID identifying this run.
    run_dir     : str   Absolute path where generated files will be written.

    Returns
    -------
    dict  Final state (same schema as multi-agent pipeline for fair comparison).
    """
    start_time = time.time()
    os.makedirs(run_dir, exist_ok=True)
    logs = [f"⚡ [Baseline] Single-LLM run started — ID: {run_id}"]

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # ── One-shot prompt ───────────────────────────────────────────────
    prompt = f"""You are a Python developer. Given the requirement below, produce
working Python source code AND pytest tests in a SINGLE response.

REQUIREMENT:
{requirement}

CODING TONE/STYLE:
{tone}

Return a JSON object where each key is a filename and each value is the
complete Python code for that file.

Rules:
  • Include 1-2 source files AND 1 test file (test file must start with "test_")
  • Python standard library only — no third-party imports
  • Complete code — no placeholders, no TODO stubs
  • Test file imports source modules directly by name (e.g. `from calculator import add`)
  • Return ONLY the raw JSON — no markdown fences

Example:
{{
  "calculator.py": "def add(a, b):\\n    return a + b\\n",
  "test_calculator.py": "from calculator import add\\n\\ndef test_add():\\n    assert add(2, 3) == 5\\n"
}}

JSON:"""

    logs.append("💬 [Baseline] Sending single prompt to GPT-4o…")

    generated_code: dict = {}
    test_code:      dict = {}

    try:
        response  = llm.invoke([HumanMessage(content=prompt)])
        all_files = _extract_json(response.content)

        generated_code = {k: v for k, v in all_files.items() if not k.startswith("test_")}
        test_code      = {k: v for k, v in all_files.items() if k.startswith("test_")}

        logs.append(f"✅ [Baseline] Received {len(all_files)} file(s) from GPT-4o")

    except Exception as exc:
        logs.append(f"❌ [Baseline] LLM error: {exc}")
        generated_code = {"main.py": f"# Code generation failed: {exc}\n"}
        test_code      = {"test_main.py": "# No tests\n"}

    # ── Sandbox ───────────────────────────────────────────────────────
    logs.append("🔬 [Baseline] Running tests and static analysis…")
    prepare_run_dir(run_dir, generated_code, test_code)

    test_results   = run_pytest(run_dir)
    lint_scores    = run_pylint(run_dir, list(generated_code.keys()))
    security_report, security_score = run_bandit(run_dir)

    status = "✅ PASSED" if test_results["passed"] else "❌ FAILED"
    logs.append(f"🧪 [Baseline] Tests: {status} ({test_results['pass_rate']}%)")

    avg_lint = (
        round(sum(lint_scores.values()) / len(lint_scores), 1)
        if lint_scores else 0.0
    )
    logs.append(f"📐 [Baseline] Lint: {avg_lint}/10")
    logs.append(f"🛡️  [Baseline] Security: {security_score}/10")

    # ── Mock GitHub ───────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    logs.append(f"⏱️  Baseline completed in {elapsed}s")

    baseline_state = {
        "run_id":          run_id,
        "run_dir":         run_dir,
        "requirement":     requirement,
        "technical_spec":  "N/A — Single-LLM baseline (no architecture planning)",
        "file_structure":  list(generated_code.keys()),
        "generated_code":  generated_code,
        "test_code":       test_code,
        "test_results":    test_results,
        "lint_scores":     lint_scores,
        "security_report": security_report,
        "security_score":  security_score,
        "review_feedback": "",
        "fix_attempts":    0,   # never loops
        "documentation":   "*(Auto-documentation not generated in Baseline mode.)*",
        "github_output":   push_to_github({
            "github_repo_url": state_github.get("github_repo_url", ""),
            "github_branch":   state_github.get("github_branch", ""),
            "github_token":    state_github.get("github_token", ""),
            "requirement":    requirement,
            "technical_spec": "Auto-generated PR from SDLC pipeline.",
            "generated_code": generated_code,
            "test_code":      test_code,
            "documentation":  "",
        }) if state_github.get("github_token") else {},
        "logs":            logs,
        "start_time":      start_time,
        "elapsed_time":    elapsed,
    }

    return baseline_state
