"""
================================================================================
# Copyright (c) 2025 Gilad Leef
#
# This software is provided for educational, research, and personal use only.
# Commercial use, resale, or distribution for profit is strictly prohibited.
# All modifications and derivative works must be distributed under the same license terms.
#
# Any disputes arising from the use of this software shall be governed by and construed in accordance with the laws of the State of Israel.
# Exclusive jurisdiction for any such disputes shall lie with the competent courts located in Israel.
================================================================================
"""

def applyConfig(module, configDict):
    """
    Apply configuration dictionary to a module.
    
    Args:
        module: Target module to configure
        configDict: Dictionary of configuration key-value pairs
    """
    for key, value in configDict.items():
        setattr(module, key, value)


def getMinipileConfig():
    """Get configuration for Minipile training."""
    return {
        "OUTPUT_DIR": "./output/minipile",
        "LOGGING_DIR": "./logs/minipile",
        "DATASET_PATH": "./data/minipile_tokenized",
        "USE_ROSA": True,
        "USE_ROSA_TRAINING": False,
        "FIRST_LAYER_GLOBAL_NO_ROSA": True,
        "WINDOW_SIZE": 512,
        "ROSA_VOCAB_SIZES": [512, 0, 0, 256, 0, 0, 128, 0],
        "ROSA_TEMPERATURE": 1.0,
        "ROSA_PAD_ID": 0,
        "ROSA_WORKERS": 8,
        "ROSA_DENSE_TABLE_MAX_ELEMS": 50_000_000,
        "HEAD_DIM": 64,
        "HIDDEN_ACT": "silu",
        "HIDDEN_SIZE": 512,
        "INITIALIZER_RANGE": 0.02,
        "INTERMEDIATE_SIZE": 2048,
        "MAX_POSITION_EMBEDDINGS": 2048,
        "MAX_WINDOW_LAYERS": None,
        "NUM_ATTENTION_HEADS": 8,
        "NUM_HIDDEN_LAYERS": 8,
        "NUM_KEY_VALUE_HEADS": 4,
        "RMS_NORM_EPS": 1e-6,
        "ROPE_THETA": 10000.0,
        "VOCAB_SIZE": 10000,
        "PAD_TOKEN_ID": 0,
        "TRAIN_EPOCHS": 1,
        "WARMUP": 0.05,
        "PER_DEVICE_TRAIN_BATCH_SIZE": 2,
        "WEIGHT_DECAY": 0.01,
        "LR_SCHEDULER_TYPE": "cosine_with_min_lr",
        "MINIR": 0.1,
        "LOGGING_STEPS": 100,
        "LEARNING_RATE": 0.002,
        "GRADIENT_ACCUMULATION_STEPS": 16,
        "SAVE_STRATEGY": "steps",
        "SAVE_STEP": 500,
        "DATALOADER_NUM_WORKERS": 4,
        "DATALOADER_PIN_MEMORY": True,
        "DATALOADER_PREFETCH_FACTOR": 4,
        "FP16": False,
        "BF16": True,
        "GRADIENT_CHECKPOINTING": False,
        "REPORT_TO": ["tensorboard"],
        "SKIP_QK_NORM": False,
    }


def getPileConfig():
    """Get configuration for full Pile training."""
    return {
        "OUTPUT_DIR": "./output/pile",
        "LOGGING_DIR": "./logs/pile",
        "DATASET_PATH": "./data/pile_tokenized",
        "USE_ROSA": True,
        "USE_ROSA_TRAINING": False,
        "FIRST_LAYER_GLOBAL_NO_ROSA": True,
        "WINDOW_SIZE": 512,
        "ROSA_VOCAB_SIZES": [8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 256, 256, 256],
        "ROSA_TEMPERATURE": 1.0,
        "ROSA_PAD_ID": 0,
        "ROSA_WORKERS": 4,
        "ROSA_DENSE_TABLE_MAX_ELEMS": 50_000_000,
        "HEAD_DIM": 128,
        "HIDDEN_ACT": "silu",
        "HIDDEN_SIZE": 2048,
        "INITIALIZER_RANGE": 0.02,
        "INTERMEDIATE_SIZE": 8192,
        "MAX_POSITION_EMBEDDINGS": 2048,
        "MAX_WINDOW_LAYERS": None,
        "NUM_ATTENTION_HEADS": 16,
        "NUM_HIDDEN_LAYERS": 24,
        "NUM_KEY_VALUE_HEADS": 8,
        "RMS_NORM_EPS": 1e-6,
        "ROPE_THETA": 10000.0,
        "VOCAB_SIZE": 50000,
        "PAD_TOKEN_ID": 0,
        "TRAIN_EPOCHS": 1,
        "WARMUP": 0.05,
        "PER_DEVICE_TRAIN_BATCH_SIZE": 8,
        "WEIGHT_DECAY": 0.01,
        "LR_SCHEDULER_TYPE": "cosine_with_min_lr",
        "MINIR": 0.1,
        "LOGGING_STEPS": 100,
        "LEARNING_RATE": 0.002,
        "GRADIENT_ACCUMULATION_STEPS": 4,
        "SAVE_STRATEGY": "steps",
        "SAVE_STEP": 5000,
        "DATALOADER_NUM_WORKERS": 4,
        "DATALOADER_PIN_MEMORY": True,
        "DATALOADER_PREFETCH_FACTOR": 4,
        "FP16": False,
        "BF16": True,
        "REPORT_TO": ["tensorboard"],
    }

