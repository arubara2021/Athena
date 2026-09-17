$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $projectRoot

Write-Host ""
Write-Host "Research Agent Installer" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

$pythonCommand = "python"

try {
    $version = & $pythonCommand --version 2>&1
    Write-Host "Detected Python: $version" -ForegroundColor Green
} catch {
    Write-Host "Python is not installed or not found in PATH." -ForegroundColor Red
    exit 1
}

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCommand -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

Write-Host "Installing core dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install -r requirements.txt

if (Test-Path "requirements-dev.txt") {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install -r requirements-dev.txt
}

Write-Host ""

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example." -ForegroundColor Yellow
        Write-Host "Please open .env and add your API keys." -ForegroundColor Yellow
    } else {
        Write-Host "No .env or .env.example found." -ForegroundColor Yellow
        Write-Host "Create a .env file with your API keys." -ForegroundColor Yellow
    }
} else {
    Write-Host ".env file detected." -ForegroundColor Green
}

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Activate the environment with: .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Verify your setup with:      python main.py config" -ForegroundColor Cyan
Write-Host ""