@echo off
rem Double-click this file to open the Blade Ball AI control panel.
rem The first time, it installs what the AI needs (a few minutes).
cd /d "%~dp0"

python -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo Python isn't installed, or Windows can't find it.
    echo.
    echo Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" on the first screen of the installer.
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)

python -c "import cv2, mss, pynput, joblib, sklearn, pandas, numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo First time: installing what the AI needs. This takes a few minutes...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing failed -- see the messages above.
        pause
        exit /b 1
    )
)

rem pythonw runs the panel without a console window.
for /f "delims=" %%p in ('python -c "import sys, pathlib; print(pathlib.Path(sys.executable).with_name('pythonw.exe'))"') do set "PYTHONW=%%p"
start "" "%PYTHONW%" app.py
