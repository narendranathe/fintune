#!/bin/bash

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
#
# Usage:
#   bash run_local_gpu.sh [--config <path>] [--skip-tests] [--skip-benchmark] [--verbose]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'  # No Color

# Configuration
CONFIG_FILE="${CONFIG_FILE:-configs/qlora_config.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"
SKIP_TESTS=false
SKIP_BENCHMARK=false
VERBOSE=false
VENV_PATH=".venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging functions
log_step() {
    echo -e "${MAGENTA}[$(date '+%H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

# Check Python version
check_python_version() {
    log_step "Checking Python version..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        return 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Found: Python $PYTHON_VERSION"

    # Check version is 3.10 or higher
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [[ $MAJOR -ge 3 ]] && [[ $MINOR -ge 10 ]]; then
        log_success "Python version is compatible"
        return 0
    else
        log_error "Python 3.10+ is required. Found: $PYTHON_VERSION"
        return 1
    fi
}

# Check CUDA availability
check_cuda() {
    log_step "Checking CUDA availability..."

    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        log_info "NVIDIA GPU found. Driver version: $CUDA_VERSION"
        log_success "CUDA environment verified"
        return 0
    else
        log_warning "NVIDIA GPU not detected. Will use CPU for training."
        return 1
    fi
}

# Setup virtual environment
setup_venv() {
    log_step "Setting up virtual environment at: $VENV_PATH"

    if [[ -d "$VENV_PATH" ]]; then
        log_info "Virtual environment already exists"
    else
        python3 -m venv "$VENV_PATH"
        log_success "Virtual environment created"
    fi

    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    log_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    log_step "Installing dependencies from requirements.txt..."

    if [[ ! -f "requirements.txt" ]]; then
        log_warning "requirements.txt not found"
        return 1
    fi

    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    log_success "Dependencies installed"
}

# Run unit tests
run_unit_tests() {
    log_step "Running unit tests..."

    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping unit tests (--skip-tests flag set)"
        return 0
    fi

    if command -v pytest &> /dev/null; then
        if pytest tests/ -v --tb=short; then
            log_success "All tests passed"
            return 0
        else
            log_warning "Some tests failed"
            return 1
        fi
    else
        log_warning "pytest not found, skipping tests"
        return 0
    fi
}

# Run sklearn benchmark
run_benchmark() {
    log_step "Running sklearn benchmark for baseline comparison..."

    if [[ "$SKIP_BENCHMARK" == "true" ]]; then
        log_info "Skipping benchmark (--skip-benchmark flag set)"
        return 0
    fi

    if [[ -f "scripts/benchmark_sklearn.py" ]]; then
        if python3 scripts/benchmark_sklearn.py; then
            log_success "Benchmark completed"
            return 0
        else
            log_warning "Benchmark reported issues"
            return 1
        fi
    else
        log_warning "benchmark_sklearn.py not found"
        return 0
    fi
}

# Run QLoRA training
run_qlora_training() {
    local config_file=$1

    log_step "Starting QLoRA fine-tuning with config: $config_file"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    if [[ -f "scripts/train_qlora.py" ]]; then
        if python3 scripts/train_qlora.py --config "$config_file"; then
            log_success "QLoRA training completed"
            return 0
        else
            log_error "QLoRA training failed"
            return 1
        fi
    else
        log_error "train_qlora.py not found"
        return 1
    fi
}

# Run evaluation
run_evaluation() {
    log_step "Running model evaluation..."

    if [[ -f "scripts/evaluate.py" ]]; then
        if python3 scripts/evaluate.py; then
            log_success "Evaluation completed"
            return 0
        else
            log_warning "Evaluation reported issues"
            return 1
        fi
    else
        log_warning "evaluate.py not found"
        return 0
    fi
}

# Run quantization
run_quantization() {
    log_step "Running model quantization..."

    if [[ -f "scripts/quantize.py" ]]; then
        if python3 scripts/quantize.py; then
            log_success "Quantization completed"
            return 0
        else
            log_warning "Quantization reported issues"
            return 1
        fi
    else
        log_warning "quantize.py not found"
        return 0
    fi
}

# Print results summary
print_summary() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}FINTUNE TRAINING PIPELINE SUMMARY${NC}"
    echo -e "${MAGENTA}========================================${NC}"

    log_info "Training pipeline execution completed."
    log_info "Results location: outputs/"
    log_info "Model checkpoint: outputs/fintune-model/"
    log_info "Logs: outputs/training.log"

    echo ""
    log_info "Next steps:"
    log_info "1. Review results in outputs/"
    log_info "2. Load model: from transformers import AutoModelForSequenceClassification"
    log_info "3. Make predictions: model.generate(...)"
    log_info "4. Deploy to production or further fine-tune"

    echo -e "${MAGENTA}========================================${NC}"
}

# Main execution function
main() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}FINTUNE LOCAL GPU TRAINING${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""

    # Step 1: Check Python
    if ! check_python_version; then
        exit 1
    fi

    # Step 2: Check CUDA
    check_cuda || true

    # Step 3: Setup virtual environment
    setup_venv

    # Step 4: Install dependencies
    if ! install_dependencies; then
        exit 1
    fi

    # Step 5: Run tests
    if ! run_unit_tests; then
        log_warning "Continuing despite test failures..."
    fi

    # Step 6: Run benchmark
    if ! run_benchmark; then
        log_warning "Continuing despite benchmark issues..."
    fi

    # Step 7: QLoRA training
    if ! run_qlora_training "$CONFIG_FILE"; then
        exit 1
    fi

    # Step 8: Evaluation
    if ! run_evaluation; then
        log_warning "Evaluation had issues, but continuing..."
    fi

    # Step 9: Quantization
    if ! run_quantization; then
        log_warning "Quantization had issues, but continuing..."
    fi

    # Print summary
    print_summary

    log_success "Training pipeline completed successfully!"
}

# Execute main function
main
