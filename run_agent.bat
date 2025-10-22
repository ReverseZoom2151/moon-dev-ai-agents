@echo off
REM Windows batch script to run agents with correct Python path
REM Usage: run_agent.bat chartanalysis_agent

if "%1"=="" (
    echo Usage: run_agent.bat [agent_name]
    echo Example: run_agent.bat chartanalysis_agent
    exit /b 1
)

set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8

echo Running agent: %1
echo Project root: %PYTHONPATH%

python "%~dp0src\agents\%1.py"
