# Autonomous SDLC Assistant

> Research + tool hybrid system for benchmarking multi-agent code generation

An AI-powered multi-agent system that autonomously transforms natural-language software requirements into production-ready Python code, complete with unit tests, security analysis, documentation, and a GitHub Pull Request.

## 🔬 Research Contribution

Empirically demonstrates that a **structured multi-agent pipeline produces measurably better code quality** than a single-LLM baseline — measured across test pass rate, lint score, and security score.

## 🏗️ Architecture

```
User Prompt → Architect → Coder → Tester → Sandbox
                                      ↑         ↓ (fail, max 3x)
                                   Reviewer ←──┘
                                      ↓ (pass)
                                   Security (bandit)
                                      ↓
                                   DocGen (README)
                                      ↓
                                   GitHub Mock (PR JSON)
```

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph |
| LLM | OpenAI GPT-4o |
| Code Execution | Python subprocess (pytest, pylint, bandit) |
| UI | Gradio 6.x |
| Metrics | Excel (openpyxl) |
| GitHub | Mock → PyGithub (swappable) |

## 🚀 Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your OpenAI API Key
copy .env.template .env
# Edit .env and add your key

# 3. Run the app
python app.py
# Open http://localhost:7860
```

## 🎯 Pipeline Modes

- **Multi-Agent** — Full 6-agent LangGraph pipeline with self-healing loop
- **Baseline (Single LLM)** — One GPT-4o prompt, no iteration
- **Both (Benchmark)** — Runs both and shows side-by-side metrics comparison

## 📊 Metrics Tracked (per run → saved to Excel)

- Test Pass Rate (%)
- Avg Lint Score (/10, via pylint)
- Security Score (/10, via bandit)
- Self-Healing Loop Count
- Lines of Code Generated
- Time to Code (seconds)

## 📁 Project Structure

```
├── app.py                    # Gradio UI
├── agents/
│   ├── architect.py          # Plans file structure
│   ├── coder.py              # Generates Python code
│   ├── tester.py             # Writes pytest tests
│   ├── reviewer.py           # Self-healing: diagnoses failures
│   ├── security.py           # bandit scan + LLM annotation
│   └── doc_gen.py            # Generates README
├── core/
│   ├── graph.py              # LangGraph pipeline
│   ├── baseline.py           # Single-LLM baseline
│   ├── sandbox.py            # pytest / pylint / bandit runner
│   └── state.py              # Shared TypedDict state
├── integrations/
│   └── github.py             # PyGithub PR creation (mock fallback)
├── metrics/
│   ├── tracker.py            # Excel read/write
│   └── benchmark_data.xlsx   # Auto-created (git-ignored)
└── output/                   # Generated files, per run (git-ignored)
```

## 🔑 Key Performance Metrics

1. **Code Correctness** — Unit test pass rate
2. **Development Velocity** — Time-to-PR, intervention rate
3. **System Robustness** — Self-correction success rate
4. **Security** — bandit HIGH/MEDIUM issue count

## 📄 License

Released under the [MIT License](LICENSE).

---
*Built with LangGraph + GPT-4o + Gradio*
