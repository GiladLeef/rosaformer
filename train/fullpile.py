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

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import model
from utils import setupMultiproc, createTokenizer, prepareData, showCuda, showConfig, countParams
from config import applyConfig, getPileConfig


"""
================================================================================
Dataset Paths
================================================================================
"""

cacheDir = "./data/pile_cache"
tokPath = "./tokenizer/pile.json"
dataPath = "./data/pile_tokenized"

tokVocab = 50000
maxLen = 2048

useStream = True
maxSamples = None


"""
================================================================================
Main Training
================================================================================
"""

def main():
    """Main training pipeline."""
    print("="*80)
    print("Rosaformer Model - Full Pile Training")
    print("="*80)
    
    showCuda()
    
    print("\n" + "!"*80)
    print("WARNING: The Pile dataset is ~825GB uncompressed")
    print(f"Streaming mode: {useStream}")
    if maxSamples:
        print(f"Training limited to: {maxSamples:,} samples")
    else:
        print("Training on full dataset (this will take a very long time)")
    print("!"*80 + "\n")
    
    applyConfig(model, getPileConfig())
    model.DATASET_PATH = dataPath
    model.USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    print("\nStep 1: Preparing tokenizer...")
    tokenizer = createTokenizer(
        savePath=tokPath,
        dataName="monology/pile-uncopyrighted",
        splitSpec="train",
        cacheDir=cacheDir,
        vocabSize=tokVocab,
        useStream=True,
        maxExamples=100000
    )
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    model.VOCAB_SIZE = len(tokenizer)
    model.PAD_TOKEN_ID = tokenizer.pad_token_id
    
    print("\nStep 2: Preparing dataset...")
    tokDataset = prepareData(
        tokenizer=tokenizer,
        dataName="monology/pile-uncopyrighted",
        splitSpec="train",
        cacheDir=cacheDir,
        savePath=dataPath,
        maxLen=maxLen,
        useStream=useStream,
        maxSamples=maxSamples
    )
    if not useStream:
        print(f"Tokenized dataset size: {len(tokDataset)} examples")
    else:
        print("Dataset prepared in streaming mode")
    
    showConfig(model)
    
    paramCount = countParams(model)
    print(f"Estimated model parameters: ~{paramCount/1e6:.1f}M")
    print()
    
    print("Step 3: Starting training...")
    print("(Training will be handled by model.main())\n")
    
    model.main()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Model saved to: {model.OUTPUT_DIR}")
    print(f"Logs saved to: {model.LOGGING_DIR}")


if __name__ == "__main__":
    setupMultiproc()
    main()

