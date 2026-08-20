import os
import time
import uuid
import json
from datetime import datetime

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from agents.architect import architect_node
from agents.coder     import coder_node
from agents.tester    import tester_node
from agents.reviewer  import reviewer_node
from agents.security  import security_node
from agents.doc_gen   import doc_gen_node
from core.sandbox     import prepare_run_dir, run_pytest, run_pylint, run_bandit
from core.baseline    import run_baseline_pipeline
from integrations.github import push_to_github
from metrics.tracker  import save_run_metrics, load_all_metrics, get_metrics_file_path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUSTOM_CSS = """
body, .gradio-container { background: #0d0d15 !important; }
.main-title {
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.9em !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.stepper { display: flex; flex-direction: column; gap: 4px; padding: 4px 0; }
.step-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 11px; border-radius: 8px;
    border: 1px solid #1a1a2e; background: #0f0f1c;
    transition: all 0.3s ease;
}
.step-idle    { border-color: #1a1a2e; }
.step-running {
    background: rgba(124,58,237,0.10); border-color: #7c3aed;
    animation: pulse-s 1.4s infinite;
}
.step-done  { background: rgba(16,185,129,0.07); border-color: rgba(16,185,129,0.4); }
.step-skip  { opacity: 0.45; }
.step-error { background: rgba(239,68,68,0.07); border-color: #ef4444; }
.step-marker { font-size: 15px; min-width: 22px; text-align: center; padding-top: 1px; }
.step-label  { font-size: 12px; font-weight: 600; color: #c4c4e4; line-height: 1.3; }
.step-desc   { font-size: 10px; color: #3d3d5c; margin-top: 1px; }
.step-running .step-label { color: #a78bfa; }
.step-done    .step-label { color: #6ee7b7; }
@keyframes pulse-s {
    0%,100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
    50%      { box-shadow: 0 0 8px 2px rgba(124,58,237,0.25); }
}
.status-banner {
    padding: 9px 14px; border-radius: 8px; font-size: 13px;
    font-weight: 600; margin-bottom: 6px; border: 1px solid transparent;
}
.sb-idle  { background:#12121f; color:#3d3d5c; border-color:#1a1a2e; }
.sb-run   { background:rgba(124,58,237,0.10); color:#a78bfa; border-color:#7c3aed; }
.sb-done  { background:rgba(16,185,129,0.08); color:#6ee7b7; border-color:#10b981; }
.sb-error { background:rgba(239,68,68,0.08);  color:#fca5a5; border-color:#ef4444; }
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 8px 0; }
.metric-card {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
    border: 1px solid rgba(124,58,237,0.3); border-radius: 10px;
    padding: 14px 10px; text-align: center;
}
.metric-card .m-val   { font-size: 1.8em; font-weight: 700; color: #a78bfa; line-height: 1.1; }
.metric-card .m-sub   { font-size: 10px; color: #818cf8; margin-top: 2px; }
.metric-card .m-label { font-size: 10px; color: #4b5563; margin-top: 4px; }
.winner-badge {
    display: inline-block; background: linear-gradient(90deg, #10b981, #059669);
    color: white; border-radius: 20px; padding: 2px 10px; font-size: 10px; font-weight: 700; margin-left: 8px;
}
.log-box textarea, .test-box textarea { font-family: 'Fira Code', 'Cascadia Code', monospace !important; font-size: 12px !important; background: #080812 !important; color: #94a3b8 !important; }
"""

_STEPS = [
    ("architect", "🏗️",  "Architect",  "Analyse requirements & plan files"),
    ("coder",     "💻",  "Coder",      "Generate Python source code"),
    ("tester",    "🧪",  "Tester",     "Write pytest unit tests"),
    ("sandbox",   "🔬",  "Sandbox",    "Run tests + pylint locally"),
    ("reviewer",  "🔄",  "Reviewer",   "Diagnose failures → fix plan"),
    ("security",  "🛡️",  "Security",  "bandit scan + LLM analysis"),
    ("doc_gen",   "📝",  "DocGen",     "Generate README"),
]

_MARKER = {"done": "✅", "running": "⏳", "skip": "⏭️", "error": "❌"}

def _stepper_html(statuses: dict) -> str:
    items = []
    for key, icon, label, desc in _STEPS:
        st  = statuses.get(key, "idle")
        mk  = _MARKER.get(st, "◯")
        items.append(
            f'<div class="step-item step-{st}">'
            f'<span class="step-marker">{mk}</span>'
            f'<div><div class="step-label">{icon} {label}</div>'
            f'<div class="step-desc">{desc}</div></div>'
            f'</div>'
        )
    return '<div class="stepper">' + "".join(items) + "</div>"

_INIT_STEPPER = _stepper_html({})

def _banner(text: str, kind: str = "idle") -> str:
    cls = {"idle": "sb-idle", "run": "sb-run", "done": "sb-done", "error": "sb-error"}.get(kind, "sb-idle")
    return f'<div class="status-banner {cls}">{text}</div>'

def _metrics_html(state: dict) -> str:
    tr  = state.get("test_results", {})
    ls  = state.get("lint_scores", {})
    avg = round(sum(ls.values()) / len(ls), 1) if ls else None
    sec = state.get("security_score", None)
    fx  = state.get("fix_attempts", 0)
    pc  = tr.get("passed_count", 0)
    fc  = tr.get("failed_count", 0)
    pr  = tr.get("pass_rate", 0.0)
    cards = [
        (f"{pr:.1f}%", f"{pc} tests automatically generated & passed", "Code Verification"),
        (f"{avg}/10" if avg is not None else "—", "via pylint", "Avg Lint Score"),
        (f"{sec}/10" if sec is not None else "—", "via bandit", "Security Score"),
        (str(fx), "bugs automatically fixed", "Self-Healing Hooks"),
    ]
    rows = "".join(f'<div class="metric-card"><div class="m-val">{v}</div><div class="m-sub">{s}</div><div class="m-label">{l}</div></div>' for v, s, l in cards)
    return f'<div class="metric-row">{rows}</div>'

def _comparison_html(b: dict, m: dict) -> str:
    def _row(label, bv, mv, hi=True):
        try:
            bn = float(str(bv).replace("%","").split("/")[0])
            mn = float(str(mv).replace("%","").split("/")[0])
            bw = (bn > mn) if hi else (bn < mn)
            mw = (mn > bn) if hi else (mn < bn)
        except Exception:
            bw = mw = False
        bb = '<span class="winner-badge">🏆</span>' if bw else ""
        mb = '<span class="winner-badge">🏆</span>' if mw else ""
        return f"<tr><td style='padding:8px 12px;color:#9ca3af'>{label}</td><td style='padding:8px 12px;text-align:center;color:#a78bfa'>{bv}{bb}</td><td style='padding:8px 12px;text-align:center;color:#6ee7b7'>{mv}{mb}</td></tr>"
    def _m(s):
        tr = s.get("test_results", {}); ls = s.get("lint_scores", {})
        return {
            "pass":  f"{tr.get('pass_rate',0):.1f}%",
            "lint":  f"{round(sum(ls.values())/len(ls),1) if ls else 0.0}/10",
            "sec":   f"{s.get('security_score',0.0)}/10",
            "fix":   s.get("fix_attempts", 0),
            "time":  f"{s.get('elapsed_time',0.0):.1f}s",
        }
    if not b or not m: return ""
    bm, mm = _m(b), _m(m)
    rows = _row("Test Pass Rate", bm["pass"], mm["pass"]) + _row("Avg Lint Score", bm["lint"], mm["lint"]) + _row("Security Score", bm["sec"], mm["sec"]) + _row("Self-Heal Loops", bm["fix"], mm["fix"], hi=False) + _row("Time to Code", bm["time"], mm["time"], hi=False)
    return f"<h4 style='color:#e2e8f0;margin:8px 0'>⚡ Baseline vs 🤖 Multi-Agent</h4><div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse;background:#1a1a2e;border-radius:10px;overflow:hidden'><thead><tr style='background:#2d1b69'><th style='padding:10px 12px;text-align:left;color:#e2e8f0'>Metric</th><th style='padding:10px 12px;text-align:center;color:#a78bfa'>⚡ Baseline</th><th style='padding:10px 12px;text-align:center;color:#6ee7b7'>🤖 Multi-Agent</th></tr></thead><tbody>{rows}</tbody></table></div>"

def _fmt_code(src: dict, tests: dict) -> str:
    parts = []
    if src:
        parts += ["# " + "═"*60, "#  📂  SOURCE FILES", "# " + "═"*60, ""]
        for fname, code in src.items(): parts += [f"# ── {fname} " + "─"*(55-len(fname)), code.strip(), ""]
    if tests:
        parts += ["", "# " + "═"*60, "#  🧪  TEST FILES", "# " + "═"*60, ""]
        for fname, code in tests.items(): parts += [f"# ── {fname} " + "─"*(55-len(fname)), code.strip(), ""]
    return "\n".join(parts) if parts else "# No code generated yet."

def _fmt_plan(spec: str, files: list) -> str:
    if not spec: return "*Waiting for Architect agent…*"
    return f"## 🏗️ Technical Specification\n\n{spec}\n\n---\n\n## 📋 Planned Files\n\n" + "\n".join(f"- `{f}`" for f in (files or []))

def _fmt_test_out(tr: dict) -> str:
    if not tr: return "Test output will appear here after the Sandbox runs…"
    header = f"{'═'*60}\n  PYTEST RESULTS  ·  {'✅ PASSED' if tr.get('passed') else '❌ FAILED'}\n  Passed: {tr.get('passed_count',0)}   Failed: {tr.get('failed_count',0)}   Pass rate: {tr.get('pass_rate',0.0)}%\n{'═'*60}\n\n"
    return header + tr.get("stdout", "") + ("\n\nSTDERR:\n" + tr.get("stderr", "") if tr.get("stderr", "").strip() else "")

def _ts() -> str: return datetime.now().strftime("%H:%M:%S")

def _init(req: str, tone: str, run_id: str, run_dir: str) -> dict:
    return {
        "run_id": run_id, "run_dir": run_dir, "requirement": req, "tone": tone,
        "technical_spec": "", "file_structure": [], "generated_code": {}, "test_code": {},
        "test_results": {}, "lint_scores": {}, "security_report": "", "security_score": 0.0,
        "review_feedback": "", "fix_attempts": 0, "documentation": "", "github_output": {},
        "logs": [f"[{_ts()}] 🚀 Pipeline started — run ID: {run_id}"],
        "start_time": time.time(), "elapsed_time": 0.0,
    }

def run_plan_stage(requirement: str, tone: str):
    """Stage 1: Run Baseline (hidden) and Architect, then Pause."""
    if not requirement.strip():
        yield tuple(list(_emit_plan(["❌ Please enter a requirement."], {}, ("❌ Please enter a requirement.", "error"))) + [gr.update(visible=False), gr.update(visible=False), {}, {}])
        return
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        yield tuple(list(_emit_plan(["❌ OPENAI_API_KEY not found in .env — please add it and restart."], {}, ("❌ API key missing.", "error"))) + [gr.update(visible=False), gr.update(visible=False), {}, {}])
        return

    run_id  = str(uuid.uuid4())[:8]
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    b_id  = run_id + "-b"
    b_dir = os.path.join(OUTPUT_DIR, b_id)
    
    ma_id = run_id + "-ma"
    ma_dir = os.path.join(OUTPUT_DIR, ma_id)
    os.makedirs(ma_dir, exist_ok=True)

    state = _init(requirement, tone, ma_id, ma_dir)
    logs = list(state["logs"])

    # Baseline (runs in background essentially, but we wait for it here so we have data)
    logs.append(f"[{_ts()}] ⚡ Running Baseline (GPT-4o single prompt) in background…")
    yield tuple(list(_emit_plan(logs, {}, ("⚡ Running Baseline behind the scenes…", "run"))) + [gr.update(visible=False), gr.update(visible=False), b_store if "b_store" in locals() else {}, state])
    baseline_st = run_baseline_pipeline(requirement, tone, {}, b_id, b_dir)
    save_run_metrics(baseline_st, "Baseline")
    logs.append(f"[{_ts()}] ✅ Baseline complete.")

    statuses = {}
    statuses["architect"] = "running"
    logs.append(f"[{_ts()}] 🏗️  Architect → planning architecture…")
    yield tuple(list(_emit_plan(logs, statuses, ("🏗️ Architect analysing requirements…", "run"))) + [gr.update(visible=False), gr.update(visible=False), baseline_st if "baseline_st" in locals() else {}, state])

    state.update(architect_node(state))
    logs = list(state["logs"])
    statuses["architect"] = "done"

    spec_text = state.get("technical_spec", "")
    file_list = state.get("file_structure", [])
    logs.append(f"[{_ts()}] ✅ Architect Plan ready. PAUSING for human approval.")
    
    ret = list(_emit_plan(logs, statuses, ("✅ Architect done. Please approve the plan to continue.", "done"), plan=_fmt_plan(spec_text, file_list)))
    ret[5] = gr.update(visible=True) # btn_approve becomes visible
    ret.extend([gr.update(visible=True), gr.update(visible=True), baseline_st, state]) # Add feedback, replan, stores
    yield tuple(ret)


def run_replan_stage(state: dict, feedback: str, baseline_st: dict):
    if not feedback.strip():
        yield tuple(list(_emit_plan(["⚠️ Please enter feedback."], {}, ("⚠️ Feedback empty.", "error"), state=state, baseline=baseline_st))[:6] + [gr.update(visible=True), gr.update(visible=True), baseline_st, state])
        return
    
    state["requirement"] += f"\n\nUSER FEEDBACK ON PREVIOUS PLAN:\n{feedback}"
    logs = list(state.get("logs", []))
    statuses = {"architect": "running"}
    logs.append(f"[{_ts()}] 🔄 Architect → revising plan based on feedback…")
    yield tuple(list(_emit_plan(logs, statuses, ("🔄 Architect revising plan…", "run"), state=state, baseline=baseline_st))[:6] + [gr.update(visible=True), gr.update(visible=True), baseline_st, state])
    
    state.update(architect_node(state))
    logs = list(state["logs"])
    statuses["architect"] = "done"

    spec_text = state.get("technical_spec", "")
    file_list = state.get("file_structure", [])
    logs.append(f"[{_ts()}] ✅ Architect Plan revised. PAUSING for human approval.")
    
    ret = list(_emit_plan(logs, statuses, ("✅ Revised plan ready. Approve to continue.", "done"), plan=_fmt_plan(spec_text, file_list), state=state, baseline=baseline_st))
    ret[5] = gr.update(visible=True) # btn_approve visible
    ret.extend([gr.update(visible=True, value=""), gr.update(visible=True), baseline_st, state]) # added plan_feedback and btn_replan
    yield tuple(ret)

def _emit_plan(logs, statuses, banner, plan="", state=None, baseline=None):
    # Returns 6 base UI items (log, stepper, banner, plan, run_vis, approve_vis)
    # The caller must manually append feedback_vis, replan_vis, bl_store, state_store to match 10 outputs.
    return ("\n".join(logs), _stepper_html(statuses), _banner(f"{banner[0]}", banner[1]), plan or "*Waiting*", gr.update(visible=False), gr.update(visible=False))

def run_execution_stage(state: dict, baseline_st: dict):
    """Stage 2: Resume from approve."""
    logs = list(state.get("logs", []))
    statuses = {"architect": "done"}
    spec_text = state.get("technical_spec", "")
    file_list = state.get("file_structure", [])
    ma_dir = state["run_dir"]

    def _emit(st, code="", test_out="", security="", readme="", cmp="", df=None, m_h="", sum_md="", sum_vis=False):
        return ("\n".join(logs), _stepper_html(statuses), _banner(f"Running Swarm...", "run"), _fmt_plan(spec_text, file_list), code or "# Waiting", test_out or "Waiting", security or "*Waiting*", readme or "*Waiting*", cmp, df if df is not None else pd.DataFrame(), m_h, gr.update(value=sum_md, visible=sum_vis), gr.update(visible=False), gr.update(visible=False), st, gr.update(visible=False))

    MAX_LOOPS = 3
    for loop_i in range(MAX_LOOPS + 1):
        is_retry = state.get("review_feedback", "")
        statuses["coder"] = "running"
        logs.append(f"[{_ts()}] " + (f"🔄 Re-generating code (attempt {loop_i+1})…" if is_retry else "💻 Generating code…"))
        yield _emit(state)
        
        state.update(coder_node(state))
        logs = list(state["logs"])
        statuses["coder"] = "done"
        src = state.get("generated_code", {})

        statuses["tester"] = "running"
        logs.append(f"[{_ts()}] 🧪 Tester → writing pytest unit tests…")
        yield _emit(state, code=_fmt_code(src, {}))
        
        state.update(tester_node(state))
        logs = list(state["logs"])
        statuses["tester"] = "done"
        tests = state.get("test_code", {})

        statuses["sandbox"] = "running"
        logs.append(f"[{_ts()}] 🔬 Sandbox → writing files to disk, running pytest + pylint…")
        yield _emit(state, code=_fmt_code(src, tests))

        prepare_run_dir(ma_dir, src, tests)
        test_results = run_pytest(ma_dir)
        lint_scores  = run_pylint(ma_dir, list(src.keys()))
        state["test_results"] = test_results
        state["lint_scores"]  = lint_scores
        logs = list(state.get("logs", logs))

        pass_str = "✅ PASSED" if test_results["passed"] else "❌ FAILED"
        logs.append(f"[{_ts()}] 🔬 pytest: {pass_str} {test_results['pass_rate']}%")
        state["logs"] = logs
        statuses["sandbox"] = "done"

        yield _emit(state, code=_fmt_code(src, tests), test_out=_fmt_test_out(test_results), m_h=_metrics_html(state))

        if test_results["passed"] or loop_i >= MAX_LOOPS - 1:
            statuses.setdefault("reviewer", "skip")
            break

        statuses["reviewer"] = "running"
        logs.append(f"[{_ts()}] 🔄 Reviewer → diagnosing failures (loop {loop_i+1})…")
        yield _emit(state, code=_fmt_code(src, tests), test_out=_fmt_test_out(test_results), m_h=_metrics_html(state))

        state.update(reviewer_node(state))
        logs = list(state["logs"])
        statuses["reviewer"] = "done"

    statuses["security"] = "running"
    logs.append(f"[{_ts()}] 🛡️ Security → scanning…")
    yield _emit(state, code=_fmt_code(src, tests), test_out=_fmt_test_out(state["test_results"]), m_h=_metrics_html(state))
    state.update(security_node(state))
    logs = list(state["logs"])
    statuses["security"] = "done"

    statuses["doc_gen"] = "running"
    logs.append(f"[{_ts()}] 📝 DocGen → generating README…")
    yield _emit(state, code=_fmt_code(src, tests), test_out=_fmt_test_out(state["test_results"]), security=state.get("security_report",""), m_h=_metrics_html(state))
    state.update(doc_gen_node(state))
    logs = list(state["logs"])
    statuses["doc_gen"] = "done"

    elapsed = round(time.time() - state["start_time"], 1)
    state["elapsed_time"] = elapsed
    logs.append(f"[{_ts()}] ⏱️ Pipeline complete in {elapsed}s")
    state["logs"] = logs
    save_run_metrics(state, "Multi-Agent")

    # Generate Final Summary via LLM
    logs.append(f"[{_ts()}] 📊 Generating final analysis…")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    summary_prompt = f"""Write a brief Executive Summary comparing the Single-LLM Baseline vs the Multi-Agent Swarm.
Baseline produced {baseline_st.get('test_results',{}).get('pass_rate',0)}% pass rate.
Swarm produced {state.get('test_results',{}).get('pass_rate',0)}% pass rate after {state.get('fix_attempts',0)} self-healing loops.
What did we achieve? What human review is logically needed before deployment?"""
    try:
        final_summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
    except Exception as e:
        final_summary = "Summary generation failed."

    # Final Output
    ret = list(_emit(
        state, code=_fmt_code(src, tests), test_out=_fmt_test_out(state["test_results"]),
        security=state.get("security_report",""), readme=state.get("documentation",""),
        cmp=_comparison_html(baseline_st, state), df=pd.DataFrame(load_all_metrics()), m_h=_metrics_html(state),
        sum_md=f"### 🏆 Final Analysis\n\n{final_summary}", sum_vis=True
    ))
    ret[2] = _banner("✅ Pipeline complete!", "done") 
    ret[13] = gr.update(visible=True) # The Run button becomes visible again
    yield tuple(ret)

def push_github_manual(state: dict, repo: str, branch: str, token: str):
    if not state: return "Wait for pipeline to finish."
    state["github_repo_url"] = repo
    state["github_branch"] = branch
    state["github_token"] = token
    res = push_to_github(state)
    if "error" in res:
        return f"❌ Error: {res['error']}"
    return f"✅ Deployed to GitHub! [View PR]({res.get('pr_url','')})"


with gr.Blocks(title="Autonomous SDLC Assistant") as demo:
    gr.HTML("<div style='text-align:center;padding:22px 0 14px'><div class='main-title'>🤖 Autonomous Self-Healing Code Engine</div><p style='color:#4b5563;margin:5px 0 0;font-size:13px'>Stop babysitting chatbots. Provide a requirement, and watch a full AI engineering pod design, write, test, debug, and deploy 100% verified code autonomously.</p></div>")

    state_store = gr.State()
    bl_store = gr.State()

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=290):
            with gr.Tabs():
                with gr.TabItem("Run"):
                    req_input = gr.Textbox(label="", placeholder="Requirement...", lines=8)
                    tone_dropdown = gr.Dropdown(choices=["Beginner / CS101 (Simple & Commented)", "Standard Academic Assignment", "Hackathon (Fast & Hacky)", "Research Level (Strict & Formal)"], value="Standard Academic Assignment", label="🎨 Coding Tone")
                    
                    run_btn = gr.Button("▶ Plan Architecture", variant="secondary")
                    plan_feedback = gr.Textbox(visible=False, label="Architect Feedback", placeholder="E.g., Please split logic into 3 files instead of 1...", lines=2)
                    with gr.Row():
                        btn_replan = gr.Button("🔄 Revise Plan", variant="secondary", visible=False)
                        btn_approve = gr.Button("✅ Approve Plan & Run Swarm", variant="primary", visible=False)

                    stepper_html = gr.HTML(value=_INIT_STEPPER)

                with gr.TabItem("GitHub"):
                    gr.HTML("<div style='font-size:13px;color:#6b7280;margin-bottom:12px;'>Configure manual GitHub push.</div>")
                    github_repo_url = gr.Textbox(label="Repo URL")
                    github_branch = gr.Textbox(label="Branch Name", value="sdlc-bot-update")
                    github_token = gr.Textbox(label="PAT", type="password")
                    deploy_btn = gr.Button("🚀 Deploy PR to GitHub")
                    github_status = gr.Markdown()

        with gr.Column(scale=3):
            banner_html  = gr.HTML(value=_banner("Ready.", "idle"))
            summary_box = gr.Markdown(visible=False, elem_id="summary_box")
            metrics_html = gr.HTML(value="")

            with gr.Tabs():
                with gr.TabItem("🏗️ Plan"): plan_output = gr.Markdown("*Waiting*")
                with gr.TabItem("💻 Code"): code_output = gr.Code(language="python", interactive=False)
                with gr.TabItem("🧪 Tests"): test_output = gr.Textbox(lines=22, interactive=False, elem_classes=["log-box"])
                with gr.TabItem("🛡️ Security"): security_output = gr.Markdown("*Waiting*")
                with gr.TabItem("📄 README"): readme_output = gr.Markdown("*Waiting*")
                with gr.TabItem("📊 Baseline vs. Swarm Benchmark"):
                    cmp_html = gr.HTML()
                    df_metrics = gr.DataFrame(label="All Runs", wrap=True)
                with gr.TabItem("⚡ Log"): log_output = gr.Textbox(lines=22, interactive=False, elem_classes=["log-box"])
                
    run_btn.click(fn=run_plan_stage, inputs=[req_input, tone_dropdown], outputs=[log_output, stepper_html, banner_html, plan_output, run_btn, btn_approve, plan_feedback, btn_replan, bl_store, state_store])
    btn_replan.click(fn=run_replan_stage, inputs=[state_store, plan_feedback, bl_store], outputs=[log_output, stepper_html, banner_html, plan_output, run_btn, btn_approve, plan_feedback, btn_replan, bl_store, state_store])
    btn_approve.click(fn=run_execution_stage, inputs=[state_store, bl_store], outputs=[log_output, stepper_html, banner_html, plan_output, code_output, test_output, security_output, readme_output, cmp_html, df_metrics, metrics_html, summary_box, plan_feedback, btn_replan, state_store, run_btn])
    deploy_btn.click(fn=push_github_manual, inputs=[state_store, github_repo_url, github_branch, github_token], outputs=[github_status])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
