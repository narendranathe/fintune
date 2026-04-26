# ================================================================
# FinTune - Complete Training Pipeline (Benchmark + QLoRA + Results)
# ================================================================
# This script does EVERYTHING:
#   1. Creates Python venv and installs dependencies
#   2. Runs pytest (guardrails, data, model config tests)
#   3. Runs sklearn baseline benchmark (CPU, ~30 seconds)
#   4. Runs QLoRA fine-tuning on GPU (~15 min on T4)
#   5. Runs evaluation and saves metrics
#   6. Saves all results to outputs/
# ================================================================
# Usage:
#   cd C:\Users\naren\projects\fintune_project\fintune
#   powershell -ExecutionPolicy Bypass -File scripts\RUN_EVERYTHING.ps1
# ================================================================

$ErrorActionPreference = "Continue"
$projectDir = "C:\Users\naren\projects\fintune_project\fintune"
Set-Location $projectDir

function Write-Step($num, $total, $msg) {
    Write-Host ""
    Write-Host "[$num/$total] $msg" -ForegroundColor Cyan
    Write-Host ("-" * 60) -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  FinTune - Full Training Pipeline" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

$totalSteps = 7

# ---- STEP 1: Python venv ----
Write-Step 1 $totalSteps "Setting up Python environment"

if (-not (Test-Path ".venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate venv
$activateScript = ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    Write-Host "  Venv activated" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Could not activate venv, using system Python" -ForegroundColor Yellow
}

# ---- STEP 2: Install dependencies ----
Write-Step 2 $totalSteps "Installing dependencies"

# Check if torch is already installed
$torchInstalled = python -c "import torch; print(torch.__version__)" 2>$null
if ($torchInstalled) {
    Write-Host "  PyTorch $torchInstalled already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing PyTorch (trying CUDA 12.1 first)..." -ForegroundColor Yellow
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  CUDA 12.1 failed, trying CUDA 11.8 (smaller download)..." -ForegroundColor Yellow
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  CUDA installs failed, installing CPU-only PyTorch..." -ForegroundColor Yellow
            pip install torch torchvision
        }
    }
}

Write-Host "  Installing remaining requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Fix datasets version - v4.x dropped script-based dataset support
$dsVersion = python -c "import datasets; v=datasets.__version__; print('old' if int(v.split('.')[0]) >= 3 else 'ok')" 2>$null
if ($dsVersion -eq "old") {
    Write-Host "  Downgrading datasets library (v4.x breaks financial_phrasebank)..." -ForegroundColor Yellow
    pip install "datasets>=2.19.0,<3.0.0"
}

Write-Host "  Dependencies installed" -ForegroundColor Green

# ---- STEP 3: Verify GPU ----
Write-Step 3 $totalSteps "Checking GPU availability"

python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  No GPU detected - will use CPU')"

$gpuAvailable = python -c "import torch; print(torch.cuda.is_available())"

# ---- STEP 4: Run tests ----
Write-Step 4 $totalSteps "Running tests"

python -m pytest tests/test_data.py tests/test_guardrails.py tests/test_model.py tests/test_serve.py -v --tb=short 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  All tests passed" -ForegroundColor Green
} else {
    Write-Host "  Some tests failed (continuing anyway)" -ForegroundColor Yellow
}

# ---- STEP 5: Sklearn baseline benchmark ----
Write-Step 5 $totalSteps "Running sklearn baseline benchmark (CPU)"

python -m src.benchmark 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Benchmark complete - results in outputs/benchmark_results.json" -ForegroundColor Green
} else {
    Write-Host "  Benchmark failed" -ForegroundColor Red
}

# ---- STEP 6: QLoRA Fine-Tuning ----
Write-Step 6 $totalSteps "Running QLoRA fine-tuning"

if ($gpuAvailable -eq "True") {
    Write-Host "  GPU detected - running Mistral-7B QLoRA training..." -ForegroundColor Yellow
    Write-Host "  This will take ~15 minutes on a T4, longer on smaller GPUs" -ForegroundColor DarkGray
    python -m src.train --config configs/qlora_config.yaml 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  QLoRA training complete!" -ForegroundColor Green

        # Run evaluation
        Write-Host "  Running evaluation on test set..." -ForegroundColor Yellow
        python -m src.evaluate --model-path outputs/fintune-financial --dataset test 2>&1

        # Run quantization
        Write-Host "  Quantizing model for inference..." -ForegroundColor Yellow
        python -m src.quantize --model-path outputs/fintune-financial --output-path outputs/fintune-financial-quantized --quant-type 4bit 2>&1
    } else {
        Write-Host "  QLoRA training failed. Check GPU memory (need 4GB+ VRAM)" -ForegroundColor Red
        Write-Host "  Trying with DistilBERT CPU config as fallback..." -ForegroundColor Yellow
        python -m src.train --config configs/qlora_distilbert_cpu.yaml 2>&1
    }
} else {
    Write-Host "  No GPU detected - skipping Mistral-7B QLoRA" -ForegroundColor Yellow
    Write-Host "  Running DistilBERT CPU config instead..." -ForegroundColor Yellow
    python -m src.train --config configs/qlora_distilbert_cpu.yaml 2>&1
}

# ---- STEP 7: Collect and display results ----
Write-Step 7 $totalSteps "Collecting results"

# Create a summary results file
python -c @"
import json, os
from pathlib import Path

results = {'benchmark': None, 'training': None}

# Load benchmark results
bench_path = Path('outputs/benchmark_results.json')
if bench_path.exists():
    with open(bench_path) as f:
        results['benchmark'] = json.load(f)

# Check for training metrics
train_dir = Path('outputs/fintune-financial')
if train_dir.exists():
    # Look for trainer_state.json
    state_files = list(train_dir.rglob('trainer_state.json'))
    if state_files:
        with open(state_files[0]) as f:
            state = json.load(f)
        if 'log_history' in state:
            # Get best metrics
            evals = [e for e in state['log_history'] if 'eval_f1_macro' in e]
            if evals:
                best = max(evals, key=lambda x: x.get('eval_f1_macro', 0))
                results['training'] = {
                    'best_f1_macro': best.get('eval_f1_macro'),
                    'best_accuracy': best.get('eval_accuracy'),
                    'best_precision': best.get('eval_precision_macro'),
                    'best_recall': best.get('eval_recall_macro'),
                }

# Save combined results
with open('outputs/final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print('=' * 70)
print('FINAL RESULTS SUMMARY')
print('=' * 70)

if results['benchmark']:
    print('\nSKLEARN BASELINES:')
    for name, m in results['benchmark'].get('models', {}).items():
        print(f'  {name}: Accuracy={m["accuracy"]:.4f}, Macro F1={m["macro_f1"]:.4f}')

if results['training']:
    t = results['training']
    print(f'\nQLORA FINE-TUNED MODEL:')
    print(f'  Macro F1:  {t["best_f1_macro"]:.4f}')
    print(f'  Accuracy:  {t["best_accuracy"]:.4f}')
    if t.get('best_precision'):
        print(f'  Precision: {t["best_precision"]:.4f}')
    if t.get('best_recall'):
        print(f'  Recall:    {t["best_recall"]:.4f}')
    print(f'\n  >> Use this for your resume: {t["best_f1_macro"]*100:.1f}% macro F1')
else:
    print('\nNo QLoRA training results found (GPU training may not have run)')

print()
print('All results saved to: outputs/final_results.json')
print('Benchmark details:    outputs/benchmark_results.json')
print('Confusion matrices:   outputs/confusion_matrix_*.png')
print('Classification rpts:  outputs/classification_report_*.txt')
print('=' * 70)
"@

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Pipeline Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Results directory: $projectDir\outputs\" -ForegroundColor White
Write-Host ""
Write-Host "  What to do next:" -ForegroundColor Yellow
Write-Host "  1. Check outputs\final_results.json for your actual metrics" -ForegroundColor White
Write-Host "  2. Update resume bullet with real F1 score" -ForegroundColor White
Write-Host "  3. Push to GitHub:" -ForegroundColor White
Write-Host "     cd C:\Users\naren\projects\fintune_project" -ForegroundColor Gray
Write-Host "     .\PUSH_TO_GITHUB.ps1" -ForegroundColor Gray
Write-Host ""
