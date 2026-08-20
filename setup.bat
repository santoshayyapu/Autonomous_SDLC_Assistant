@echo off
setlocal
echo ==============================================
echo   Autonomous SDLC Assistant - Setup Engine
echo ==============================================
echo.

:: Check for .env file
if not exist .env (
    echo [INFO] Creating placeholder .env file...
    echo OPENAI_API_KEY=your_openai_api_key_here > .env
    echo GITHUB_TOKEN=your_optional_github_token_here >> .env
    echo [WARN] Please configure your .env file with your actual API keys!
) else (
    echo [INFO] .env file already exists.
)

:: Check for Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.10+
    pause
    exit /b
)

:: Ensure venv exists
if not exist venv (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: Check if requirements file exists
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found. Are you in the root project folder?
    pause
    exit /b
)

:: Setup and boot
echo [INFO] Activating virtual environment and verifying dependencies...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul
pip install -r requirements.txt

echo.
echo ==============================================
echo            Starting the UI Server...
echo ==============================================
python app.py

pause
