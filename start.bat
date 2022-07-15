cd %~dp0
call venv\Scripts\activate.bat
python wsgi.py
pause