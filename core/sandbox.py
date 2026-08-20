"""Sandbox: safely executes generated code in a subprocess.

Provides:
  - prepare_run_dir()  — write code/test files to disk
  - run_pytest()       — execute pytest, parse pass/fail
  - run_pylint()       — lint source files, extract score
  - run_bandit()       — security scan, compute 0-10 score
"""

import os
import re
import json
import sys
import subprocess
from typing import Dict, Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_run_dir(
    run_dir: str,
    code_files: Dict[str, str],
    test_files: Dict[str, str],
) -> None:
    """Write all generated source and test files to *run_dir*."""
    os.makedirs(run_dir, exist_ok=True)
    for filename, content in {**code_files, **test_files}.items():
        filepath = os.path.join(run_dir, filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)


# ── pytest ────────────────────────────────────────────────────────────────────

def run_pytest(run_dir: str) -> Dict:
    """Run pytest inside *run_dir* and return a structured result dict."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", ".", "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=run_dir,
        )
        output = result.stdout + result.stderr

        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
        errors = int(m.group(1)) if (m := re.search(r"(\d+) error", output)) else 0

        total = passed + failed + errors
        pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0

        return {
            "passed": result.returncode == 0,
            "pass_rate": pass_rate,
            "passed_count": passed,
            "failed_count": failed + errors,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:1000],
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False, "pass_rate": 0.0,
            "passed_count": 0, "failed_count": 1,
            "stdout": "⚠️ Tests timed out after 90 seconds.", "stderr": "",
        }
    except Exception as exc:
        return {
            "passed": False, "pass_rate": 0.0,
            "passed_count": 0, "failed_count": 1,
            "stdout": str(exc), "stderr": "",
        }


# ── pylint ────────────────────────────────────────────────────────────────────

def run_pylint(run_dir: str, filenames: list) -> Dict[str, float]:
    """Lint each *non-test* .py file in *filenames* and return filename→score."""
    scores: Dict[str, float] = {}

    for filename in filenames:
        if filename.startswith("test_") or not filename.endswith(".py"):
            continue
        filepath = os.path.join(run_dir, filename)
        if not os.path.exists(filepath):
            continue
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pylint", filename,
                    "--score=yes",
                    "--disable=C0114,C0115,C0116,W0401",  # skip missing-docstring, wildcard-import
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=run_dir,
            )
            combined = result.stdout + result.stderr
            m = re.search(r"rated at ([\d.]+)/10", combined)
            scores[filename] = float(m.group(1)) if m else 5.0
        except Exception:
            scores[filename] = 5.0

    return scores


# ── bandit ────────────────────────────────────────────────────────────────────

def run_bandit(run_dir: str) -> Tuple[str, float]:
    """
    Run bandit over *run_dir* (excluding test files).
    Returns (report_json_str, security_score_0_to_10).
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "bandit",
                "-r", ".",
                "-f", "json",
                "--exclude", "./*test*,./__pycache__",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=run_dir,
        )

        report_text = result.stdout or "{}"

        try:
            data = json.loads(report_text)
            issues = data.get("results", [])
            high   = sum(1 for i in issues if i.get("issue_severity") == "HIGH")
            medium = sum(1 for i in issues if i.get("issue_severity") == "MEDIUM")
            score  = max(0.0, round(10.0 - high * 2.0 - medium * 0.5, 1))
        except (json.JSONDecodeError, KeyError):
            score = 9.0  # Default: assume clean when bandit output is unparseable

        return report_text, score

    except Exception as exc:
        return f'{{"error": "{exc}"}}', 8.0
