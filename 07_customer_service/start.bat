@echo off
chcp 65001 >nul
title 智能客服启动器
cd /d %~dp0

echo ============================================
echo   智能客服系统 - 启动脚本
echo ============================================
echo.

rem 1. 启动 Ollama（embedding 服务）
echo [1/3] 检查 Ollama 服务...
netstat -ano | findstr :11434 >nul 2>&1
if errorlevel 1 (
    echo   Ollama 未运行，正在启动...
    start "Ollama" "F:\Ollama\ollama.exe" serve
    timeout /t 5 /nobreak >nul
    echo   Ollama 已启动。
) else (
    echo   Ollama 已在运行。
)

rem 2. 检查 embedding 模型
echo [2/3] 检查 embedding 模型...
curl -s http://localhost:11434/api/tags | findstr "qwen3-embedding:0.6b" >nul 2>&1
if errorlevel 1 (
    echo   模型 qwen3-embedding:0.6b 不存在，正在拉取（首次较慢）...
    ollama pull qwen3-embedding:0.6b
) else (
    echo   embedding 模型已就绪。
)

rem 3. 启动 Web 服务
echo [3/3] 启动 Web 服务...
echo   访问地址: http://127.0.0.1:8000
echo   按 Ctrl+C 停止
echo.
call ..\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

pause
