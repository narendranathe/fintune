"""Post-training quantization for inference optimization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def quantize_for_inference(
    model_path: str,
    output_path: str,
    quant_type: str = "4bit",
) -> None:
    """Load merged model and re-quantize for efficient inference.

    Supports 4-bit (NF4) and 8-bit quantization via bitsandbytes.
    The quantized model loads faster and uses ~4x less VRAM than fp16.

    Args:
        model_path: Path to merged (full-precision) model.
        output_path: Where to save the quantized model.
        quant_type: '4bit' for NF4 quantization, '8bit' for INT8.
    """
    logger.info("Quantizing %s with %s precision...", model_path, quant_type)

    if quant_type == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_type == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}. Use '4bit' or '8bit'.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    # Report size reduction
    orig_size = sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors"))
    quant_size = sum(f.stat().st_size for f in output.rglob("*.safetensors"))
    ratio = orig_size / max(quant_size, 1)

    logger.info("Quantized model saved to %s (%.1fx compression)", output, ratio)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--quant-type", default="4bit", choices=["4bit", "8bit"])
    args = parser.parse_args()
    quantize_for_inference(args.model_path, args.output_path, args.quant_type)
