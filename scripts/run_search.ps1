    param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Topic,

    [string]$Goal = "Learn from basics to advanced",
    [string]$Level = "beginner to advanced",
    [int]$MaxResults = 10,
    [int]$Limit = 5,
    [string[]]$Platform = @(),
    [switch]$NoLlmRanking,
    [switch]$LlmExpansion,
    [switch]$LlmLearningPath,
    [switch]$Json,
    [switch]$Markdown,
    [switch]$Quiet,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found." -ForegroundColor Red
    Write-Host "Run scripts\install.ps1 first." -ForegroundColor Yellow
    exit 1
}

$arguments = @(
    "main.py",
    "search",
    $Topic,
    "--goal", $Goal,
    "--level", $Level,
    "--max-results", $MaxResults,
    "--limit", $Limit
)

foreach ($p in $Platform) {
    $arguments += "--platform"
    $arguments += $p
}

if ($NoLlmRanking)       { $arguments += "--no-llm-ranking" }
if ($LlmExpansion)       { $arguments += "--llm-expansion" }
if ($LlmLearningPath)    { $arguments += "--llm-learning-path" }
if ($Json)               { $arguments += "--json" }
if ($Markdown)           { $arguments += "--markdown" }
if ($Quiet)              { $arguments += "--quiet" }
if ($Open)               { $arguments += "--open" }

& $venvPython @arguments