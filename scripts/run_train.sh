#!/bin/bash
set -euo pipefail

echo "=== FinTune Training Pipeline ==="
echo "Step 1: Running tests..."
pytest tests/ -v --tb=short

echo ""
echo "Step 2: Starting QLoRA fine-tuning..."
python -m src.train --config configs/qlora_config.yaml

echo ""
echo "Step 3: Evaluating model..."
python -m src.evaluate --model-path outputs/fintune-financial --dataset test

echo ""
echo "Step 4: Quantizing for inference..."
python -m src.quantize --model-path outputs/fintune-financial --output-path outputs/fintune-quantized --quant-type 4bit

echo ""
echo "=== Training pipeline complete ==="
echo "Start serving with: uvicorn src.serve:app --host 0.0.0.0 --port 8000"
