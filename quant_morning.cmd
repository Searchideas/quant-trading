@echo off
cd /d "c:\Users\taiji\Desktop\Python Project\Quant Learning Journey\quant-trading"
set PYTHONUNBUFFERED=1
set LOG=%USERPROFILE%\quant-morning.log

powershell -NoProfile -Command "if (@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*broker.telegram serve*' }).Count -gt 0) { exit 1 } else { exit 0 }"
if errorlevel 1 exit /b 0

:again
echo %DATE% %TIME% starting >> "%LOG%"
python -m broker.telegram serve >> "%LOG%" 2>&1
echo %DATE% %TIME% stopped, retry in 15s >> "%LOG%"
timeout /t 15 /nobreak >nul
goto again
