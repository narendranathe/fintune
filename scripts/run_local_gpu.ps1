# FinTune Local GPU Training Pipeline
# Prerequisites: Python 3.10+, CUDA 12.1+, ~16GB VRAM
#
# This script orchestrates the complete training workflow:
# 1. Environment verification
# 2. Virtual environment setup
# 3. Dependency installation
# 4. Unit tests execution
# 5. Benchmark comparison
# 6. QLoRA fine-tuning
# 7. Model evaluation
# 8. Quantization
# 9. Results summary

param(
    [string]$ConfigFile = "configs/qlora_config.yaml",
    [string]$ProjectRoot = ".",
    [switch]$SkipTests = $false,
    [switch]$SkipBenchmark = $false,
    [switch]$VerboseOutput = $false
)

# Colors for console output
$Colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
    Step = "Magenta"
}

function Write-Step {
    param([string]$Message)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $Message" -ForegroundColor $Colors.Step
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Success
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Error
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Warning
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Info
}

# Check Python version
function Check-PythonVersion {
    Write-Step "Checking Python version..."

    try {
        $PythonVersion = python --version 2>&1
        Write-Info "Found: $PythonVersion"

        # Extract version number
        if ($PythonVersion -match "3\.1[0-9]") {
            Write-Success "Python version is compatible"
            return $true
        } else {
            Write-Error-Custom "Python 3.10+ is required. Found: $PythonVersion"
            return $false
        }
    } catch {
        Write-Error-Custom "Python not found or error occurred: $_"
        return $false
    }
}

# Check CUDA availability
function Check-CUDA {
    Write-Step "Checking CUDA availability..."

    try {
        $CudaVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Info "NVIDIA GPU found. Driver version: $CudaVersion"
            Write-Success "CUDA environment verified"
            return $true
        } else {
            Write-Warning-Custom "NVIDIA GPU not detected. Will use CPU for training."
            return $false
        }
    } catch {
        Write-Warning-Custom "Could not verify CUDA: $_"
        return $false
    }
}

# Create virtual environment
function Setup-VirtualEnvironment {
    param([string]$VenvPath = ".venv")

    Write-Step "Setting up virtual environment at: $VenvPath"

    if (Test-Path $VenvPath) {
        Write-Info "Virtual environment already exists"
        return $true
    }

    try {
        python -m venv $VenvPath
        Write-Success "Virtual environment created"

        # Activate virtual environment
        & ".\$VenvPath\Scripts\Activate.ps1"
        Write-Success "Virtual environment activated"
        return $true
    } catch {
        Write-Error-Custom "Failed to create virtual environment: $_"
        return $false
    }
}

# Install dependencies
function Install-Dependencies {
    Write-Step "Installing dependencies from requirements.txt..."

    try {
        if (Test-Path "requirements.txt") {
            pip install --upgrade pip setuptools wheel
            pip install -r requirements.txt
            Write-Success "Dependencies installed"
            return $true
        } else {
            Write-Warning-Custom "requirements.txt not found"
            return $false
        }
    } catch {
        Write-Error-Custom "Failed to install dependencies: $_"
        return $false
    }
}

# Run unit tests
function Run-UnitTests {
    Write-Step "Running unit tests..."

    try {
        if (-not $SkipTests) {
            pytest tests/ -v --tb=short

            if ($LASTEXITCODE -eq 0) {
                Write-Success "All tests passed"
                return $true
            } else {
                Write-Warning-Custom "Some tests failed"
                return $false
            }
        } else {
            Write-Info "Skipping unit tests (--SkipTests flag set)"
            return $true
        }
    } catch {
        Write-Error-Custom "Error running tests: $_"
        return $false
    }
}

# Run sklearn benchmark
function Run-Benchmark {
    Write-Step "Running sklearn benchmark for baseline comparison..."

    try {
        if (-not $SkipBenchmark) {
            python scripts/benchmark_sklearn.py

            if ($LASTEXITCODE -eq 0) {
                Write-Success "Benchmark completed"
                return $true
            } else {
                Write-Warning-Custom "Benchmark reported issues"
                return $false
            }
        } else {
            Write-Info "Skipping benchmark (--SkipBenchmark flag set)"
            return $true
        }
    } catch {
        Write-Error-Custom "Error running benchmark: $_"
        return $false
    }
}

# Run QLoRA training
function Run-QLoRA-Training {
    param([string]$ConfigFile)

    Write-Step "Starting QLoRA fine-tuning with config: $ConfigFile"

    try {
        if (-not (Test-Path $ConfigFile)) {
            Write-Error-Custom "Config file not found: $ConfigFile"
            return $false
        }

        python scripts/train_qlora.py --config $ConfigFile

        if ($LASTEXITCODE -eq 0) {
            Write-Success "QLoRA training completed"
            return $true
        } else {
            Write-Error-Custom "QLoRA training failed with exit code: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Error-Custom "Error during QLoRA training: $_"
        return $false
    }
}

# Run evaluation
function Run-Evaluation {
    Write-Step "Running model evaluation..."

    try {
        python scripts/evaluate.py

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Evaluation completed"
            return $true
        } else {
            Write-Warning-Custom "Evaluation reported issues"
            return $false
        }
    } catch {
        Write-Error-Custom "Error during evaluation: $_"
        return $false
    }
}

# Run quantization
function Run-Quantization {
    Write-Step "Running model quantization..."

    try {
        python scripts/quantize.py

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Quantization completed"
            return $true
        } else {
            Write-Warning-Custom "Quantization reported issues"
            return $false
        }
    } catch {
        Write-Error-Custom "Error during quantization: $_"
        return $false
    }
}

# Print results summary
function Print-Summary {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor $Colors.Step
    Write-Host "FINTUNE TRAINING PIPELINE SUMMARY" -ForegroundColor $Colors.Step
    Write-Host "========================================" -ForegroundColor $Colors.Step

    Write-Info "Training pipeline execution completed."
    Write-Info "Results location: outputs/"
    Write-Info "Model checkpoint: outputs/fintune-model/"
    Write-Info "Logs: outputs/training.log"

    Write-Host ""
    Write-Info "Next steps:"
    Write-Info "1. Review results in outputs/"
    Write-Info "2. Load model: from transformers import AutoModelForSequenceClassification"
    Write-Info "3. Make predictions: model.generate(...)"
    Write-Info "4. Deploy to production or further fine-tune"

    Write-Host "========================================" -ForegroundColor $Colors.Step
}

# Main execution
function Main {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor $Colors.Step
    Write-Host "FINTUNE LOCAL GPU TRAINING" -ForegroundColor $Colors.Step
    Write-Host "========================================" -ForegroundColor $Colors.Step
    Write-Host ""

    # Step 1: Check Python
    if (-not (Check-PythonVersion)) {
        exit 1
    }

    # Step 2: Check CUDA
    $HasGPU = Check-CUDA

    # Step 3: Setup virtual environment
    if (-not (Setup-VirtualEnvironment)) {
        exit 1
    }

    # Step 4: Install dependencies
    if (-not (Install-Dependencies)) {
        exit 1
    }

    # Step 5: Run tests
    if (-not (Run-UnitTests)) {
        Write-Warning-Custom "Continuing despite test failures..."
    }

    # Step 6: Run benchmark
    if (-not (Run-Benchmark)) {
        Write-Warning-Custom "Continuing despite benchmark issues..."
    }

    # Step 7: QLoRA training
    if (-not (Run-QLoRA-Training $ConfigFile)) {
        exit 1
    }

    # Step 8: Evaluation
    if (-not (Run-Evaluation)) {
        Write-Warning-Custom "Evaluation had issues, but continuing..."
    }

    # Step 9: Quantization
    if (-not (Run-Quantization)) {
        Write-Warning-Custom "Quantization had issues, but continuing..."
    }

    # Print summary
    Print-Summary

    Write-Success "Training pipeline completed successfully!"
}

# Execute main function
Main
