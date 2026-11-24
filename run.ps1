# PowerShell Runner for Agentic AI Data Scientist
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path (Split-Path -Parent $ScriptDir) "venv\Scripts\python.exe"

Write-Host "Starting Agentic AI Data Scientist studio..." -ForegroundColor Cyan
& $VenvPython -m streamlit run (Join-Path $ScriptDir "app.py")
