@echo off
setlocal

cd /d "%~dp0"

rem 国内网络默认镜像；可通过同名环境变量覆盖
if not defined HF_ENDPOINT set "HF_ENDPOINT=https://hf-mirror.com"
if not defined PIP_INDEX_URL set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo ==============================================
echo   企业知识问答 Agent - 一键准备环境
echo   1) 创建 .venv（如果不存在）
echo   2) 安装 requirements.txt
echo   3) 下载/加载向量模型并检查索引
echo ==============================================
echo   HF  镜像: %HF_ENDPOINT%
echo   pip 镜像: %PIP_INDEX_URL%
echo ==============================================
echo.

if exist ".venv\Scripts\python.exe" goto install

where py >nul 2>nul
if not errorlevel 1 (
    echo [1/3] 使用 py -3.13 创建虚拟环境...
    py -3.13 -m venv .venv
) else (
    echo [1/3] 使用 python 创建虚拟环境...
    python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 虚拟环境创建失败，请确认已安装 Python 3.13。
    pause
    exit /b 1
)

:install
echo [2/3] 安装依赖（首次可能较慢，已使用 pip 镜像）...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或 Python 版本。
    echo 可尝试手动指定其他镜像，例如：
    echo   set PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
    pause
    exit /b 1
)

echo [3/3] 准备向量模型与索引（已使用 HF 镜像）...
".venv\Scripts\python.exe" run.py setup
if errorlevel 1 (
    echo [错误] 向量模型准备失败。
    echo 可尝试换用其他 HF 端点，例如：
    echo   set HF_ENDPOINT=https://huggingface.co
    pause
    exit /b 1
)

echo.
echo 准备完成。双击 start.bat 启动，或运行：
echo   .venv\Scripts\python.exe run.py demo --case normal
echo.
pause
endlocal