"""Tests for model configuration and LoRA setup."""

import pytest
from peft import LoraConfig, TaskType
from src.model import build_lora_config, build_quantization_config


class TestLoRAConfig:
    def test_default_config(self):
        config = build_lora_config()
        assert isinstance(config, LoraConfig)
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.task_type == TaskType.SEQ_CLS
        assert config.bias == "none"

    def test_custom_rank(self):
        config = build_lora_config(r=8, lora_alpha=16)
        assert config.r == 8
        assert config.lora_alpha == 16

    def test_custom_targets(self):
        targets = ["q_proj", "v_proj"]
        config = build_lora_config(target_modules=targets)
        assert list(config.target_modules) == targets


class TestQuantizationConfig:
    def test_default_4bit(self):
        config = build_quantization_config()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_disable_double_quant(self):
        config = build_quantization_config(use_double_quant=False)
        assert config.bnb_4bit_use_double_quant is False
