@echo off
cd /d "E:\Python projects\Voice\whisper_ptt"
call venv\Scripts\activate.bat
start /b pythonw whisper_ptt_gui.py
