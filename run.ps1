# ─────────────────────────────────────────────────────────────
#  Autonomous SDLC Assistant — Launcher
#  Usage: Just double-click this file, or run:  .\run.ps1
# ─────────────────────────────────────────────────────────────

$venvPython = ".\venv\Scripts\python.exe"
$appFile    = "app.py"

# Check venv exists
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv not found. Create it first:" -ForegroundColor Red
    Write-Host "  python -m venv venv" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Check .env exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found." -ForegroundColor Yellow
    Write-Host "  Copy .env.template to .env and add your OPENAI_API_KEY." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   🤖  Autonomous SDLC Assistant              ║" -ForegroundColor Cyan
Write-Host "  ║   LangGraph + GPT-4o · Multi-Agent Pipeline  ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Starting server on http://localhost:7860 ..." -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

& $venvPython $appFile
