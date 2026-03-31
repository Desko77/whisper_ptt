echo Starting WhisperPTT...
cd /d "E:\Python projects\Voice\whisper_ptt"
echo Current dir: %CD%
call venv\Scripts\activate.bat
echo Python:
python --version
echo Running script...
python whisper_ptt_cuda.py
echo Exit code: %ERRORLEVEL%
pause
