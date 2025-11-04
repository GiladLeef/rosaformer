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
import multiprocessing as mp
from pathlib import Path
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


"""
================================================================================
Multiprocessing Setup
================================================================================
"""

def setupMultiproc():
    """Setup multiprocessing with spawn method for ROSA pool compatibility."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


"""
================================================================================
Tokenizer Creation
================================================================================
"""

def createTokenizer(
    savePath: str,
    dataName: str,
    splitSpec: str,
    cacheDir: str,
    vocabSize: int,
    useStream: bool = False,
    maxExamples: int = None
):
    """
    Create and train a BPE tokenizer on a dataset.
    
    Args:
        savePath: Path to save/load tokenizer
        dataName: HuggingFace dataset name
        splitSpec: Dataset split specification
        cacheDir: Cache directory for dataset
        vocabSize: Tokenizer vocabulary size
        useStream: Whether to use streaming mode
        maxExamples: Maximum examples for training (streaming only)
        
    Returns:
        PreTrainedTokenizerFast: Trained tokenizer
    """
    tokFile = Path(savePath)
    
    if tokFile.exists():
        print(f"Loading existing tokenizer from {savePath}")
        return PreTrainedTokenizerFast(tokenizer_file=str(tokFile))
    
    print(f"Training new tokenizer on {dataName}...")
    
    if useStream:
        print(f"Loading {dataName} in streaming mode...")
        dataset = load_dataset(dataName, split=splitSpec, streaming=True, cache_dir=cacheDir)
    else:
        print(f"Loading {dataName}...")
        dataset = load_dataset(dataName, split=splitSpec, cache_dir=cacheDir)
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocabSize,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
    )
    
    if useStream:
        def batchIter(batchSize=1000):
            """Iterator for streaming dataset."""
            batch = []
            count = 0
            for example in dataset:
                batch.append(example["text"])
                count += 1
                
                if len(batch) >= batchSize:
                    yield batch
                    batch = []
                
                if maxExamples and count >= maxExamples:
                    break
            
            if batch:
                yield batch
    else:
        def batchIter(batchSize=1000):
            """Iterator for non-streaming dataset."""
            for i in range(0, len(dataset), batchSize):
                yield dataset[i:i + batchSize]["text"]
    
    print("Training tokenizer...")
    tokenizer.train_from_iterator(batchIter(), trainer=trainer)
    
    tokFile.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokFile))
    
    wrappedTok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    
    print(f"Tokenizer trained and saved to {savePath}")
    return wrappedTok


"""
================================================================================
Dataset Tokenization
================================================================================
"""

def makeTokenizer(tokenizer, maxLen: int):
    """
    Create a tokenization function for dataset mapping.
    
    Args:
        tokenizer: Tokenizer to use
        maxLen: Maximum sequence length
        
    Returns:
        Function that tokenizes examples
    """
    def tokenizeFunc(examples):
        """Tokenize text and create causal language modeling format."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=maxLen,
            padding="max_length",
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    return tokenizeFunc


def prepareData(tokenizer, dataName: str, splitSpec: str, cacheDir: str, savePath: str, maxLen: int, useStream: bool = False, maxSamples: int = None):
    """
    Load and tokenize a dataset.
    
    Args:
        tokenizer: Tokenizer to use
        dataName: Dataset name
        splitSpec: Split specification
        cacheDir: Cache directory
        savePath: Path to save tokenized dataset
        maxLen: Maximum sequence length
        useStream: Whether to use streaming mode
        maxSamples: Maximum training samples (optional)
        
    Returns:
        Tokenized dataset
    """
    tokPath = Path(savePath)
    
    if tokPath.exists() and not useStream:
        print(f"Loading tokenized dataset from {savePath}")
        return load_from_disk(savePath)
    
    print(f"Loading {dataName} dataset...")
    
    if useStream:
        print("Using streaming mode")
        dataset = load_dataset(dataName, split=splitSpec, streaming=True, cache_dir=cacheDir)
        
        if maxSamples is not None:
            print(f"Limiting to {maxSamples:,} samples")
            dataset = dataset.take(maxSamples)
    else:
        dataset = load_dataset(dataName, split=splitSpec, cache_dir=cacheDir)
        
        if maxSamples is not None:
            print(f"Limiting to {maxSamples:,} samples")
            dataset = dataset.select(range(maxSamples))
        
        print(f"Dataset loaded: {len(dataset)} examples")
    
    tokenizeFunc = makeTokenizer(tokenizer, maxLen)
    
    print("Tokenizing dataset...")
    
    if useStream:
        tokDataset = dataset.map(
            tokenizeFunc,
            batched=True,
            remove_columns=["text", "meta"] if "meta" in dataset.column_names else ["text"],
        )
    else:
        tokDataset = dataset.map(
            tokenizeFunc,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            num_proc=8,
        )
        
        print(f"Saving tokenized dataset to {savePath}")
        tokPath.parent.mkdir(parents=True, exist_ok=True)
        tokDataset.save_to_disk(savePath)
    
    return tokDataset


"""
================================================================================
Display Functions
================================================================================
"""

def showCuda():
    """Print CUDA availability and configuration."""
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        gpuMem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpuMem:.1f} GB")
        print(f"BF16 support: {torch.cuda.is_bf16_supported()}")
    else:
        print("CUDA not available, training on CPU (will be slow)")


def showConfig(modelModule):
    """
    Print model configuration summary.
    
    Args:
        modelModule: The model module with configuration variables
    """
    print("\n" + "="*80)
    print("Model Configuration Summary")
    print("="*80)
    print(f"Model Architecture:")
    print(f"  Hidden Size: {modelModule.HIDDEN_SIZE}")
    print(f"  Num Layers: {modelModule.NUM_HIDDEN_LAYERS}")
    print(f"  Num Heads: {modelModule.NUM_ATTENTION_HEADS}")
    print(f"  KV Heads: {modelModule.NUM_KEY_VALUE_HEADS}")
    print(f"  Vocab Size: {modelModule.VOCAB_SIZE}")
    print(f"\nROSA Configuration:")
    print(f"  Enabled: {modelModule.USE_ROSA}")
    print(f"  Window Size: {modelModule.WINDOW_SIZE}")
    print(f"  ROSA Vocab Size: {modelModule.ROSA_VOCAB_SIZES}")
    print(f"  Temperature: {modelModule.ROSA_TEMPERATURE}")
    print(f"  Workers: {modelModule.ROSA_WORKERS}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {modelModule.TRAIN_EPOCHS}")
    print(f"  Batch Size: {modelModule.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Gradient Accumulation: {modelModule.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective Batch Size: {modelModule.PER_DEVICE_TRAIN_BATCH_SIZE * modelModule.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning Rate: {modelModule.LEARNING_RATE}")
    print(f"  Warmup Ratio: {modelModule.WARMUP}")
    print(f"  Output Dir: {modelModule.OUTPUT_DIR}")
    print("="*80 + "\n")


def countParams(modelModule):
    """
    Estimate model parameters based on configuration.
    
    Args:
        modelModule: The model module with configuration variables
        
    Returns:
        int: Estimated parameter count
    """
    paramCount = (
        modelModule.VOCAB_SIZE * modelModule.HIDDEN_SIZE +
        modelModule.NUM_HIDDEN_LAYERS * (
            3 * modelModule.HIDDEN_SIZE * modelModule.HIDDEN_SIZE +
            modelModule.HIDDEN_SIZE * modelModule.HIDDEN_SIZE +
            2 * modelModule.HIDDEN_SIZE * modelModule.INTERMEDIATE_SIZE +
            4 * modelModule.HIDDEN_SIZE
        ) +
        modelModule.VOCAB_SIZE * modelModule.HIDDEN_SIZE
    )
    return paramCount

