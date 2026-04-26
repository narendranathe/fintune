# FinTune — Push to GitHub and Update Portfolio
# Run this from PowerShell: .\scripts\push_to_github.ps1
# Prerequisites: git configured with GitHub credentials

$ErrorActionPreference = "Stop"

Write-Host "`n=== Step 1: Create and push FinTune repo ===" -ForegroundColor Cyan

# Navigate to project
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$fintunePath = Join-Path $projectPath "fintune"
Set-Location $fintunePath

# Init git if needed
if (-not (Test-Path ".git")) {
    git init -b main
    git config user.name "narendranathe"
    git config user.email "edaranarendranath9@gmail.com"
}

# Stage and commit
git add -A
$hasChanges = git status --porcelain
if ($hasChanges) {
    git commit -m "feat: FinTune v0.2.0 — production-grade financial NLP with QLoRA

- QLoRA fine-tuning on Mistral-7B with PEFT LoRA adapters
- 4-bit NF4 quantization via bitsandbytes (~4x VRAM reduction)
- FastAPI serving with /predict, /predict/batch, /health, /metrics
- AI guardrails: PII redaction, confidence thresholding, audit logging
- Real-time monitoring: latency histograms, drift detection, health scoring
- Self-recovery: circuit breaker, auto model reload, OOM batch reduction
- Data pipeline: streaming loader, quality validation, deduplication
- Sklearn baselines: TF-IDF + LogReg/RF/SVC benchmarks
- 35+ test cases covering all modules
- Docker + GitHub Actions CI/CD"
}

# Create repo on GitHub (requires gh CLI or manual creation)
Write-Host "`nCreating GitHub repo..." -ForegroundColor Yellow
try {
    gh repo create narendranathe/fintune --public --description "Production-grade financial NLP: QLoRA fine-tuning, 4-bit quantized inference, AI guardrails, real-time monitoring, and self-recovery" --source . --remote origin --push
    Write-Host "Repo created and pushed!" -ForegroundColor Green
} catch {
    Write-Host "gh CLI not available or repo exists. Trying git push..." -ForegroundColor Yellow
    git remote add origin https://github.com/narendranathe/fintune.git 2>$null
    git push -u origin main
    Write-Host "Pushed to existing repo!" -ForegroundColor Green
}

Write-Host "`n=== Step 2: Update Portfolio ===" -ForegroundColor Cyan

# Navigate to portfolio
$portfolioPath = Join-Path (Split-Path -Parent $projectPath) "narendranathe"
if (-not (Test-Path $portfolioPath)) {
    $portfolioPath = "C:\Users\naren\projects\narendranathe"
}
Set-Location $portfolioPath

# Create or checkout the branch
$branchExists = git branch --list "feat/portfolio-apex"
if (-not $branchExists) {
    git checkout -b feat/portfolio-apex
} else {
    git checkout feat/portfolio-apex
}

Write-Host "`nPortfolio branch ready. Please add FinTune to README.md manually or run:" -ForegroundColor Yellow
Write-Host "  git add -A && git commit -m 'feat: add FinTune project case study' && git push -u origin feat/portfolio-apex" -ForegroundColor White

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host "FinTune repo: https://github.com/narendranathe/fintune"
Write-Host "Portfolio branch: feat/portfolio-apex"
