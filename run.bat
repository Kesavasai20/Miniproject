@echo off
echo Starting FloatChat...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Streamlit app
streamlit run app.py
