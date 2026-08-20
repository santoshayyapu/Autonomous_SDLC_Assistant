"""LangGraph Multi-Agent SDLC Pipeline

Graph topology:
                        architect
                            │
                          coder ◄──────────────────┐
                            │                       │
                          tester                    │
                            │                       │(loop, max 3x)
                          sandbox                   │
                        (run tests)                 │
                     ┌────┴────┐                    │
                 (fail)       (pass)                │
                     │           │                  │
                  reviewer ──────┘──────────────────┘
                               │(pass)
                           security
                               │
                            doc_gen
                               │
                            github
                               │
                             END
"""

import os
import time

from langgraph.graph import StateGraph, END

from agents.architect import architect_node
from agents.coder     import coder_node
from agents.tester    import tester_node
from agents.reviewer  import reviewer_node
from agents.security  import security_node
from agents.doc_gen   import doc_gen_node
from core.sandbox     import prepare_run_dir, run_pytest, run_pylint
from integrations.github_mock import push_mock_github


# ── Inline nodes (use sandbox + github) ──────────────────────────────────────

def sandbox_node(state: dict) -> dict:
    """Write files to disk, run pytest + pylint, update state."""
    logs = list(state.get("logs", []))
    logs.append("🔬 [Sandbox] Writing files and running pytest + pylint…")

    run_dir = state.get("run_dir", ".")
    code    = state.get("generated_code", {})
    tests   = state.get("test_code", {})

    prepare_run_dir(run_dir, code, tests)
    test_results = run_pytest(run_dir)
    lint_scores  = run_pylint(run_dir, list(code.keys()))

    status = "✅ PASSED" if test_results["passed"] else f"❌ FAILED"
    logs.append(
        f"🧪 [Sandbox] Tests: {status} — "
        f"{test_results['passed_count']} passed / "
        f"{test_results['failed_count']} failed "
        f"({test_results['pass_rate']}%)"
    )

    avg_lint = (
        round(sum(lint_scores.values()) / len(lint_scores), 1)
        if lint_scores else 0.0
    )
    logs.append(f"📐 [Sandbox] Avg lint score: {avg_lint}/10")

    return {"test_results": test_results, "lint_scores": lint_scores, "logs": logs}


def github_node(state: dict) -> dict:
    """Create mock GitHub PR and record elapsed time."""
    elapsed = round(time.time() - state.get("start_time", time.time()), 1)
    logs = list(state.get("logs", []))
    logs.append("🚀 [GitHub] Generating mock Pull Request…")

    pr_payload = push_mock_github(state)
    logs.append(f"✅ [GitHub] Mock PR ready → {pr_payload.get('pr_url', 'N/A')}")
    logs.append(f"⏱️  Pipeline finished in {elapsed}s")

    return {"github_output": pr_payload, "elapsed_time": elapsed, "logs": logs}


# ── Routing function ──────────────────────────────────────────────────────────

def _should_fix_or_proceed(state: dict) -> str:
    """After sandbox: loop back to reviewer, or proceed to security."""
    passed      = state.get("test_results", {}).get("passed", False)
    fix_attempts = state.get("fix_attempts", 0)

    if not passed and fix_attempts < 3:
        return "fix"
    return "proceed"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph():
    """Assemble and compile the LangGraph SDLC pipeline."""
    wf = StateGraph(dict)

    wf.add_node("architect", architect_node)
    wf.add_node("coder",     coder_node)
    wf.add_node("tester",    tester_node)
    wf.add_node("sandbox",   sandbox_node)
    wf.add_node("reviewer",  reviewer_node)
    wf.add_node("security",  security_node)
    wf.add_node("doc_gen",   doc_gen_node)
    wf.add_node("github",    github_node)

    wf.set_entry_point("architect")
    wf.add_edge("architect", "coder")
    wf.add_edge("coder",     "tester")
    wf.add_edge("tester",    "sandbox")

    wf.add_conditional_edges(
        "sandbox",
        _should_fix_or_proceed,
        {"fix": "reviewer", "proceed": "security"},
    )
    wf.add_edge("reviewer", "coder")    # self-healing loop
    wf.add_edge("security", "doc_gen")
    wf.add_edge("doc_gen",  "github")
    wf.add_edge("github",   END)

    return wf.compile()


# Compile once at import time; reused across runs
_compiled_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_multiagent_pipeline(requirement: str, run_id: str, run_dir: str) -> dict:
    """
    Execute the full multi-agent SDLC pipeline.

    Parameters
    ----------
    requirement : str   Natural-language software requirement.
    run_id      : str   Short UUID identifying this run.
    run_dir     : str   Absolute path where generated files will be written.

    Returns
    -------
    dict  Final pipeline state.
    """
    os.makedirs(run_dir, exist_ok=True)

    initial_state = {
        "run_id":          run_id,
        "run_dir":         run_dir,
        "requirement":     requirement,
        "technical_spec":  "",
        "file_structure":  [],
        "generated_code":  {},
        "test_code":       {},
        "test_results":    {},
        "lint_scores":     {},
        "security_report": "",
        "security_score":  0.0,
        "review_feedback": "",
        "fix_attempts":    0,
        "documentation":   "",
        "github_output":   {},
        "logs":            [f"🚀 [Pipeline] Multi-Agent run started — ID: {run_id}"],
        "start_time":      time.time(),
        "elapsed_time":    0.0,
    }

    return _compiled_graph.invoke(initial_state)
