@echo off
echo Setting up Python virtual environment...

REM Create virtual environment
python -m venv .venv

REM Activate it
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo.
echo Environment is ready. Type `venv\Scripts\activate` to activate manually later.
pause
