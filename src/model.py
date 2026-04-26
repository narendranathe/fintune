"""QLoRA model configuration, PEFT adapter setup, and quantization."""

from __future__ import annotations

import logging

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .data import ID_TO_LABEL, LABEL_MAP, NUM_LABELS

logger = logging.getLogger(__name__)


def build_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """Create bitsandbytes quantization config for 4-bit QLoRA.

    Args:
        load_in_4bit: Enable 4-bit quantization.
        bnb_4bit_quant_type: Quantization type (nf4 or fp4).
        bnb_4bit_compute_dtype: Compute dtype for quantized layers.
        use_double_quant: Enable nested quantization for memory savings.

    Returns:
        BitsAndBytesConfig for model loading.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.bfloat16)

    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )

    logger.info(
        "Quantization config: 4bit=%s, type=%s, double_quant=%s",
        load_in_4bit,
        bnb_4bit_quant_type,
        use_double_quant,
    )
    return config


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Create LoRA adapter configuration.

    Args:
        r: LoRA rank — controls adapter capacity vs. parameter count.
        lora_alpha: Scaling factor. Higher alpha = stronger adapter signal.
        lora_dropout: Dropout on LoRA layers for regularization.
        target_modules: Which layers to attach adapters to.

    Returns:
        LoraConfig for PEFT.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    trainable_params = r * len(target_modules) * 2  # approximate
    logger.info("LoRA config: r=%d, alpha=%d, ~%d trainable params per layer", r, lora_alpha, trainable_params)
    return config


def load_model_with_qlora(
    model_name: str,
    quantization_config: BitsAndBytesConfig | None = None,
    lora_config: LoraConfig | None = None,
    device_map: str = "auto",
) -> PreTrainedModel:
    """Load base model with 4-bit quantization and attach LoRA adapters.

    Pipeline:
        1. Load base model with bitsandbytes 4-bit quantization
        2. Prepare model for k-bit training (freeze base, enable grad checkpointing)
        3. Attach PEFT LoRA adapters to target attention layers

    Args:
        model_name: HF model identifier (e.g., 'mistralai/Mistral-7B-v0.3').
        quantization_config: 4-bit quant settings. Built with defaults if None.
        lora_config: LoRA adapter settings. Built with defaults if None.
        device_map: Device placement strategy.

    Returns:
        PEFT-wrapped model ready for training.
    """
    if quantization_config is None:
        quantization_config = build_quantization_config()

    if lora_config is None:
        lora_config = build_lora_config()

    logger.info("Loading base model: %s", model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Freeze base weights and enable gradient checkpointing for memory efficiency
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )

    # Attach LoRA adapters — only these are trained
    model = get_peft_model(base_model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.info(
        "PEFT model ready: %s trainable / %s total (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )

    return model


def merge_and_save(model: PreTrainedModel, output_path: str) -> None:
    """Merge LoRA adapters into base model and save for deployment.

    Args:
        model: PEFT-wrapped model with trained adapters.
        output_path: Directory to save merged model.
    """
    logger.info("Merging LoRA adapters into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    logger.info("Merged model saved to %s", output_path)
