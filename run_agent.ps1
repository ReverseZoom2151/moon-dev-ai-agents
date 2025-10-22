# PowerShell script to run agents with correct Python path
# Usage: .\run_agent.ps1 chartanalysis_agent

param(
    [Parameter(Mandatory=$true)]
    [string]$AgentName
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Running agent: $AgentName" -ForegroundColor Green
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

python "$ProjectRoot\src\agents\$AgentName.py"
