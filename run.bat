@echo off
echo Starting Agentic AI Data Scientist...
call "%~dp0..\venv\Scripts\activate.bat"
streamlit run "%~dp0app.py"
pause
