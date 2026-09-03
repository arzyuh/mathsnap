@echo off
REM MathSnap - стартува ги двата сервиси со еден клик.
REM Секој сервис се отвора во свој прозорец - едноставно затвори го
REM прозорецот кога сакаш да го изгаснеш тој сервис.

echo Стартувам главен backend (порт 8000)...
start "MathSnap - главен backend" cmd /k "cd /d "%~dp0backend" && "C:\Users\Computer\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn app.main:app --port 8000"

echo Стартувам ракописен сервис (порт 8001, ~10-15s да се вчита)...
start "MathSnap - ракописен модел" cmd /k "cd /d "%~dp0" && ".venv_trocr\Scripts\python.exe" backend\handwriting_service\server.py"

echo.
echo Двата сервиси се стартуваат во посебни прозорци.
echo Почекај ~15 секунди, потоа отвори: http://127.0.0.1:8000
echo.
echo (Затвори ги cmd прозорците подоцна за да ги изгаснеш сервисите.)
timeout /t 12 >nul
start http://127.0.0.1:8000
