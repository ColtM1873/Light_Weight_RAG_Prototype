@echo off
setlocal

cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8010"
set "URL=http://%HOST%:%PORT%"

echo ==============================================
echo   企业知识问答 Agent - RAG + Guard 演示
echo   地址: %URL%
echo ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没有找到 .venv\Scripts\python.exe
    echo 请先按 README.md 安装依赖。
    pause
    exit /b 1
)

if not exist "run.py" (
    echo [错误] 没有找到 run.py，请在仓库根目录运行本脚本。
    pause
    exit /b 1
)

rem 端口已被占用时，直接打开浏览器（可能是之前启动的服务）。
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 端口 %PORT% 已有服务在运行，直接打开浏览器。
    start "" "%URL%"
    exit /b 0
)

echo 正在启动服务窗口...
start "RAG Guard Server" cmd /k ".venv\Scripts\python.exe run.py serve --host %HOST% --port %PORT%"

echo 等待服务就绪...
powershell -NoProfile -ExecutionPolicy Bypass -Command "for($i=0;$i -lt 80;$i++){ try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('%HOST%',%PORT%); $c.Close(); exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1" >nul

if errorlevel 1 (
    echo [错误] 服务在 40 秒内没有就绪，请查看 "RAG Guard Server" 窗口的报错。
) else (
    echo 服务已经就绪。
)

start "" "%URL%"
endlocal