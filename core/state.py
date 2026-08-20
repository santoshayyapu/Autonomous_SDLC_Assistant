"""Shared state schema for the Autonomous SDLC pipeline.

All LangGraph nodes read from and write to this TypedDict.
Using total=False so nodes only need to return the keys they update.
"""

from typing import TypedDict, List, Dict, Any, Optional


class SDLCState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────
    run_id: str           # Short UUID for this run
    run_dir: str          # Absolute path to output/<run_id>/

    # ── Input ─────────────────────────────────────────────────────────
    requirement: str      # Raw natural-language requirement from user

    # ── Architect outputs ─────────────────────────────────────────────
    technical_spec: str   # Detailed technical specification
    file_structure: List[str]  # Planned filenames (source only)

    # ── Coder outputs ─────────────────────────────────────────────────
    generated_code: Dict[str, str]   # filename → Python code

    # ── Tester outputs ────────────────────────────────────────────────
    test_code: Dict[str, str]        # test_filename → pytest code

    # ── Sandbox results ───────────────────────────────────────────────
    test_results: Dict[str, Any]     # {passed, pass_rate, passed_count, failed_count, stdout, stderr}
    lint_scores: Dict[str, float]    # filename → pylint score (0–10)

    # ── Self-healing loop ─────────────────────────────────────────────
    review_feedback: str  # Reviewer's fix instructions for the Coder
    fix_attempts: int     # How many fix loops have run (max 3)

    # ── Security Agent ────────────────────────────────────────────────
    security_report: str  # LLM-annotated bandit findings
    security_score: float # Derived 0–10 score

    # ── Documentation ─────────────────────────────────────────────────
    documentation: str    # Generated README.md content

    # ── GitHub Mock / Real ────────────────────────────────────────────
    github_repo_url: str
    github_branch: str
    github_token: str
    github_output: Dict[str, Any]  # PR payload dict / results

    # ── Tracking ──────────────────────────────────────────────────────
    logs: List[str]       # Chronological log messages (appended by each node)
    start_time: float     # time.time() at pipeline start
    elapsed_time: float   # Seconds from start to end
