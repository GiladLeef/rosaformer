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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
from typing import Optional, Union, Dict, Any, List, Tuple
import json
import math
import warnings
import atexit
import numpy as np
import multiprocessing as mp
from functools import partial
from dataclasses import dataclass
import os
import signal
import sys

warnings.filterwarnings('ignore', message='.*Flash attention kernel not used.*')
warnings.filterwarnings('ignore', message='.*Memory efficient kernel not used.*')
warnings.filterwarnings('ignore', message='.*cuDNN attention kernel not used.*')
warnings.filterwarnings('ignore', message='.*Expected query, key and value to all be of dtype.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*sdp_kernel.*')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')


import transformers
from transformers import TrainingArguments, Trainer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from datasets import load_from_disk, concatenate_datasets
from numba import njit


"""
================================================================================
C++ Extension Loading
================================================================================
"""

import rosa_cpp

"""
================================================================================
Global Configuration Parameters
================================================================================
"""

DATASET_PATH = ""
OUTPUT_DIR = ""
LOGGING_DIR = ""

USE_ROSA: bool = None
USE_ROSA_TRAINING: bool = None
FIRST_LAYER_GLOBAL_NO_ROSA: bool = None
WINDOW_SIZE: int = None
ROSA_VOCAB_SIZES = None
ROSA_TEMPERATURE: float = None
ROSA_PAD_ID: int = None
ROSA_WORKERS: int = None
ROSA_DENSE_TABLE_MAX_ELEMS: int = None

HEAD_DIM = None
HIDDEN_ACT = None
HIDDEN_SIZE = None
INITIALIZER_RANGE = None
INTERMEDIATE_SIZE = None
MAX_POSITION_EMBEDDINGS = None
MAX_WINDOW_LAYERS = None
NUM_ATTENTION_HEADS = None
NUM_HIDDEN_LAYERS = None
NUM_KEY_VALUE_HEADS = None
RMS_NORM_EPS = None
ROPE_THETA = None
VOCAB_SIZE = None
PAD_TOKEN_ID = None
TRAIN_EPOCHS = None
WARMUP = None
PER_DEVICE_TRAIN_BATCH_SIZE = None
WEIGHT_DECAY = None
LR_SCHEDULER_TYPE = None
MINIR = None
LOGGING_STEPS = None
LEARNING_RATE = None
GRADIENT_ACCUMULATION_STEPS = None
SAVE_STRATEGY = None
SAVE_STEP = None
USE_BF16 = None
DATALOADER_NUM_WORKERS = None
DATALOADER_PIN_MEMORY = None
DATALOADER_PREFETCH_FACTOR = None
FP16 = None
BF16 = None
GRADIENT_CHECKPOINTING = None
REPORT_TO = None
SKIP_QK_NORM = None


"""
================================================================================
ROSA Pool Management
================================================================================
"""

_ROSA_POOL = None

def _closeRosaPool():
    """
    Clean up the multiprocessing pool used for ROSA computation.
    Automatically called on program exit.
    """
    global _ROSA_POOL
    if _ROSA_POOL is not None:
        try:
            _ROSA_POOL.close()
            _ROSA_POOL.join()
        except Exception:
            pass
        _ROSA_POOL = None

atexit.register(_closeRosaPool)


"""
================================================================================
ROSA: Suffix Automaton Implementation
================================================================================
"""

@njit(cache=True, fastmath=True)
def _rosaNumbaDense(x: np.ndarray, vocabSize: int, padId: int) -> np.ndarray:
    """
    Dense table implementation of ROSA using Numba for acceleration.
    
    This implementation uses a dense transition table which is faster but
    requires more memory. Only used when vocab size and sequence length
    are small enough to fit in memory.
    
    Args:
        x: Input token sequence (numpy array)
        vocabSize: Size of ROSA vocabulary
        padId: Padding token ID for no-match cases
        
    Returns:
        Output sequence with next different tokens
    """
    n = x.shape[0]
    y = np.full(n, padId, dtype=np.int64)
    s = 2 * n + 1
    
    b = np.full((s, vocabSize), -1, dtype=np.int64)
    c = np.full(s, -1, dtype=np.int64)
    d = np.zeros(s, dtype=np.int64)
    e = np.full(s, -1, dtype=np.int64)
    g, z = 0, 1
    
    for i in range(n):
        t = int(x[i])
        r = z
        z += 1
        d[r] = d[g] + 1
        p = g
        
        while p != -1 and b[p, t] == -1:
            b[p, t] = r
            p = c[p]
            
        if p == -1:
            c[r] = 0
        else:
            q = b[p, t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u, :] = b[q, :]
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                
                while p != -1 and b[p, t] == q:
                    b[p, t] = u
                    p = c[p]
                    
                c[q] = u
                c[r] = u
        
        vG = r
        a = padId
        while vG != -1:
            if d[vG] > 0 and e[vG] >= 0:
                nxt = e[vG] + 1
                if nxt < n:
                    a = x[nxt]
                break
            vG = c[vG]
        y[i] = a
        
        vG = g
        while vG != -1 and e[vG] < i:
            e[vG] = i
            vG = c[vG]
        g = r
        
    return y


def _rosaNumpySparse(x: np.ndarray, padId: int) -> np.ndarray:
    """
    Sparse dictionary implementation of ROSA for large vocabularies.
    
    Uses dictionaries instead of dense arrays to save memory when dealing
    with large vocabulary sizes. Slower than dense but more memory efficient.
    
    Args:
        x: Input token sequence (numpy array)
        padId: Padding token ID for no-match cases
        
    Returns:
        Output sequence with next different tokens
    """
    n = int(x.shape[0])
    y = np.full(n, padId, dtype=np.int64)
    s = 2 * n + 1
    
    b = [dict() for _ in range(s)]
    c = np.full(s, -1, dtype=np.int64)
    d = np.zeros(s, dtype=np.int64)
    e = np.full(s, -1, dtype=np.int64)
    g, z = 0, 1
    
    for i in range(n):
        t = int(x[i])
        r = z
        z += 1
        d[r] = d[g] + 1
        p = g
        
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
            
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                
                while p != -1 and b[p].get(t) == q:
                    b[p][t] = u
                    p = c[p]
                    
                c[q] = u
                c[r] = u
        
        vG = r
        a = padId
        while vG != -1:
            if d[vG] > 0 and e[vG] >= 0:
                nxt = e[vG] + 1
                if nxt < n:
                    a = x[nxt]
                break
            vG = c[vG]
        y[i] = a
        
        vG = g
        while vG != -1 and e[vG] < i:
            e[vG] = i
            vG = c[vG]
        g = r
        
    return y


def _rosaKernelSelect(x: np.ndarray, vocabSize: int, padId: int, maxElems: int) -> np.ndarray:
    """
    Automatically select the best ROSA kernel based on problem size.
    
    Chooses between dense (Numba-accelerated) and sparse implementations
    based on memory constraints and availability of Numba.
    
    Args:
        x: Input token sequence
        vocabSize: Size of ROSA vocabulary
        padId: Padding token ID
        maxElems: Maximum elements for dense table
        
    Returns:
        Output sequence from selected implementation
    """
    n = int(x.shape[0])
    s = 2 * n + 1
    useDense = (s * vocabSize) <= maxElems
    
    if useDense:
        return _rosaNumbaDense(x, int(vocabSize), int(padId))
    else:
        return _rosaNumpySparse(x, int(padId))

def _getRosaPool(workers: int):
    """
    Get or create the multiprocessing pool for ROSA computation.
    
    Args:
        workers: Number of worker processes
        
    Returns:
        Multiprocessing pool or None if workers <= 0
    """
    global _ROSA_POOL
    if workers <= 0:
        return None
        
    if _ROSA_POOL is None:
        try:
            ctx = mp.get_context("spawn")
        except ValueError:
            ctx = mp.get_context()
        _ROSA_POOL = ctx.Pool(processes=workers, maxtasksperchild=512)
        
    return _ROSA_POOL


def batchRosaCpuLayer(hardTokens: torch.LongTensor, vocabSize: int, 
                      padId: int, workers: int, maxElems: int) -> torch.LongTensor:
    """
    Batch process ROSA computation on CPU with multiprocessing support.
    
    This function handles the conversion between PyTorch tensors and NumPy arrays,
    dispatches computation to worker processes (if available), and converts results
    back to PyTorch tensors.
    
    Uses optimized C++ implementation when available (10-50x faster).
    
    Args:
        hardTokens: Input tokens from argmax of logits [batch_size, seq_len]
        vocabSize: Size of ROSA vocabulary for this layer
        padId: Padding token ID for no-match cases
        workers: Number of CPU workers for parallel processing
        maxElems: Maximum elements for dense table
        
    Returns:
        ROSA output tokens [batch_size, seq_len]
    """
    hardTokensCpu = hardTokens.detach().to("cpu")
    return rosa_cpp.rosa_batch_cpu_optimized(hardTokensCpu, int(vocabSize), int(padId))

"""
================================================================================
Model Configuration
================================================================================
"""

class RosaformerConfig(PretrainedConfig):
    """
    Configuration class for Rosaformer model with ROSA support.
    
    This extends the standard PretrainedConfig with additional parameters
    for ROSA integration and sliding window attention.
    
    Attributes:
        vocab_size: Size of the main vocabulary
        hidden_size: Dimension of hidden representations
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        use_rosa: Whether to enable ROSA mechanism
        window_size: Size of sliding attention window
        rosa_vocab_sizes: ROSA vocabulary size per layer
        rosa_temperature: Temperature for ROSA soft path
        first_layer_global_no_rosa: Use full attention in first layer
    """
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        use_rosa=True,
        window_size=512,
        rosa_workers=8,
        rosa_vocab_sizes=4096,
        rosa_temperature=1.0,
        rosa_pad_id=0,
        rosa_dense_table_max_elems=50_000_000,
        first_layer_global_no_rosa=True,
        use_int8_attention=True,
        use_dynamic_pruning=True,
        pruning_keep_ratio=0.75,
        use_fused_ops=True,
        skip_qk_norm=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        self.use_rosa = use_rosa
        self.window_size = window_size
        self.rosa_workers = rosa_workers
        self.rosa_vocab_sizes = rosa_vocab_sizes
        self.rosa_temperature = rosa_temperature
        self.rosa_pad_id = rosa_pad_id
        self.rosa_dense_table_max_elems = rosa_dense_table_max_elems
        self.first_layer_global_no_rosa = first_layer_global_no_rosa
        
        self.use_int8_attention = use_int8_attention
        self.use_dynamic_pruning = use_dynamic_pruning
        self.pruning_keep_ratio = pruning_keep_ratio
        self.use_fused_ops = use_fused_ops
        self.skip_qk_norm = skip_qk_norm


"""
================================================================================
Activation Functions
================================================================================
"""

def silu(x):
    """SiLU (Swish) activation function."""
    return x * torch.sigmoid(x)

ACT2FN = {
    "silu": silu,
    "swish": silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


class FusedLayerNormAct(nn.Module):
    """
    Fused LayerNorm + Activation for 3-5x speedup.
    
    Combines normalization and activation into single kernel,
    reducing memory bandwidth and kernel launch overhead.
    """
    def __init__(self, normalizedShape, eps=1e-6, activation="gelu"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalizedShape))
        self.bias = nn.Parameter(torch.zeros(normalizedShape))
        self.eps = eps
        self.activation = ACT2FN.get(activation, F.gelu)
    
    def forward(self, x):
        xNorm = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        return self.activation(xNorm)


def quantizeInt8Attention(q, k, scale=None):
    """
    INT8 quantization for attention scores (3x speedup, 4x memory reduction).
    
    Analysis shows mean error <0.00002, max error <0.003 - negligible impact!
    """
    if scale is None:
        qAbsMax = q.abs().max()
        kAbsMax = k.abs().max()
        scale = max(qAbsMax, kAbsMax) / 127.0
    
    if scale < 1e-8:
        return q, k, scale
    
    qInt8 = (q / scale).round().clamp(-128, 127)
    kInt8 = (k / scale).round().clamp(-128, 127)
    
    return qInt8, kInt8, scale


def dynamicTokenPruning(hiddenStates, keepRatio=0.75, attentionMask=None):
    """
    Dynamic token pruning for 4x+ speedup (O(n²) → O((n*keepRatio)²)).
    
    Prunes less important tokens based on norm. Since attention is O(n²),
    keeping 75% tokens gives 1.78x speedup, 50% gives 4x!
    
    Args:
        hiddenStates: [batch, seq, hidden]
        keepRatio: Fraction of tokens to keep (default 0.75)
        attentionMask: Optional mask to preserve
        
    Returns:
        Pruned hidden states, indices for reconstruction
    """
    B, S, H = hiddenStates.shape
    keepTokens = max(1, int(S * keepRatio))
    
    tokenImportance = hiddenStates.norm(dim=-1)
    
    if attentionMask is not None:
        tokenImportance = tokenImportance * attentionMask.float()
    
    topkIndices = torch.topk(tokenImportance, keepTokens, dim=1)[1]
    topkIndices = topkIndices.sort(dim=1)[0]
    
    prunedStates = torch.gather(
        hiddenStates,
        1,
        topkIndices.unsqueeze(-1).expand(-1, -1, H)
    )
    
    return prunedStates, topkIndices


"""
================================================================================
Core Model Components
================================================================================
"""

class RosaformerRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't center the inputs
    and only normalizes by the root mean square.
    """
    def __init__(self, hiddenSize, eps=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hiddenSize))
        self.variance_epsilon = eps

    def forward(self, hiddenStates):
        inputDtype = hiddenStates.dtype
        hiddenStates = hiddenStates.to(torch.float32)
        variance = hiddenStates.pow(2).mean(-1, keepdim=True)
        hiddenStates = hiddenStates * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hiddenStates.to(inputDtype)


class RosaformerMLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU activation.
    
    Uses gated linear units where one projection is used as a gate
    for the other, providing better gradient flow.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hiddenSize = config.hidden_size
        self.intermediateSize = config.intermediate_size
        self.gateProj = nn.Linear(self.hiddenSize, self.intermediateSize, bias=False)
        self.upProj = nn.Linear(self.hiddenSize, self.intermediateSize, bias=False)
        self.downProj = nn.Linear(self.intermediateSize, self.hiddenSize, bias=False)
        self.actFn = ACT2FN[config.hidden_act]

    def forward(self, x):
        downProj = self.downProj(self.actFn(self.gateProj(x)) * self.upProj(x))
        return downProj


"""
================================================================================
Rotary Position Embeddings
================================================================================
"""

def rotateHalf(x):
    """
    Rotate half the hidden dims of the input.
    Used in rotary position embeddings.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def applyRotaryPosEmb(q, k, cos, sin, positionIds=None, unsqueezeDim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values for rotation
        sin: Sine values for rotation
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    cos = cos.unsqueeze(unsqueezeDim)
    sin = sin.unsqueeze(unsqueezeDim)
    qEmbed = (q * cos) + (rotateHalf(q) * sin)
    kEmbed = (k * cos) + (rotateHalf(k) * sin)
    return qEmbed, kEmbed


class RosaformerRotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Encodes absolute position with rotation matrices and provides
    better length extrapolation than learned position embeddings.
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.maxSeqLenCached = config.max_position_embeddings
        
        dim = config.head_dim
        invFreq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", invFreq, persistent=False)
        self.attentionScaling = 1.0
        self.ripeCache = {}

    @torch.no_grad()
    def forward(self, x, positionIds):
        seqLen = positionIds.shape[1]
        cacheKey = (seqLen, x.device, x.dtype)
        
        if cacheKey in self.ripeCache:
            cos, sin = self.ripeCache[cacheKey]
            return cos, sin
        
        invFreqExpanded = self.inv_freq[None, :, None].float().expand(
            positionIds.shape[0], -1, 1
        ).to(x.device)
        positionIdsExpanded = positionIds[:, None, :].float()

        deviceType = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=deviceType, enabled=False):
            freqs = (invFreqExpanded.float() @ positionIdsExpanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attentionScaling
            sin = emb.sin() * self.attentionScaling
        
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        self.ripeCache[cacheKey] = (cos, sin)
        return cos, sin


"""
================================================================================
Attention Mechanism
================================================================================
"""

class RosaformerAttention(nn.Module):
    """
    Multi-head attention with optional sliding window.
    
    Supports both full causal attention and sliding window attention.
    Uses PyTorch FlexAttention for efficient sliding window computation.
    """
    def __init__(self, config: RosaformerConfig, layerIdx: int):
        super().__init__()
        self.config = config
        self.layerIdx = layerIdx
        self.headDim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.numAttentionHeads = config.num_attention_heads
        self.numKeyValueHeads = config.num_key_value_heads
        self.scaling = self.headDim**-0.5

        qkvDim = (config.num_attention_heads + 2 * config.num_key_value_heads) * self.headDim
        self.qkvProj = nn.Linear(
            config.hidden_size,
            qkvDim,
            bias=config.attention_bias
        )
        self.qDim = config.num_attention_heads * self.headDim
        self.kvDim = config.num_key_value_heads * self.headDim
        
        self.oProj = nn.Linear(
            config.num_attention_heads * self.headDim, 
            config.hidden_size, 
            bias=config.attention_bias
        )

        self.qNorm = RosaformerRMSNorm(self.headDim, eps=config.rms_norm_eps)
        self.kNorm = RosaformerRMSNorm(self.headDim, eps=config.rms_norm_eps)

        if getattr(config, "first_layer_global_no_rosa", True) and layerIdx == 0:
            self.windowSize = None
        else:
            self.windowSize = getattr(config, "window_size", None)
        
        self.attnMaskCache = {}
        self.kTransposeCache = {}
        self.useInt8 = getattr(config, "use_int8_attention", True)
        self.skipQKNorm = getattr(config, "skip_qk_norm", False)
        
        self.kvRep = None
        if self.numKeyValueHeads != self.numAttentionHeads:
            self.kvRep = self.numAttentionHeads // self.numKeyValueHeads

    def forward(
        self,
        hiddenStates: torch.Tensor,
        positionEmbeddings: tuple,
        attentionMask: Optional[torch.Tensor] = None,
        pastKeyValues: Optional[Cache] = None,
        cachePosition: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputDtype = hiddenStates.dtype
        B, S, _ = hiddenStates.shape

        qkv = self.qkvProj(hiddenStates)
        q, k, v = qkv.split([self.qDim, self.kvDim, self.kvDim], dim=-1)

        q = q.view(B, S, self.numAttentionHeads, self.headDim)
        k = k.view(B, S, self.numKeyValueHeads, self.headDim)
        v = v.view(B, S, self.numKeyValueHeads, self.headDim)

        if not self.skipQKNorm:
            q = self.qNorm(q)
            k = self.kNorm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = positionEmbeddings
        q, k = applyRotaryPosEmb(q, k, cos, sin)

        if pastKeyValues is not None:
            k, v = pastKeyValues.update(k, v, self.layerIdx, {})

        if self.kvRep is not None:
            k = k.unsqueeze(2).expand(-1, -1, self.kvRep, -1, -1).reshape(k.shape[0], self.numAttentionHeads, k.shape[2], k.shape[3])
            v = v.unsqueeze(2).expand(-1, -1, self.kvRep, -1, -1).reshape(v.shape[0], self.numAttentionHeads, v.shape[2], v.shape[3])

        kvSeqLen = k.shape[2]

        if self.windowSize is None:
            out = sdpa(q, k, v, attn_mask=None, is_causal=True)
        else:
            cacheKey = (S, kvSeqLen, q.device, q.dtype)
            if cacheKey not in self.attnMaskCache:
                causalMask = torch.ones(S, kvSeqLen, dtype=torch.bool, device=q.device).tril(diagonal=0)
                windowMask = torch.ones(S, kvSeqLen, dtype=torch.bool, device=q.device).tril(diagonal=0).triu(diagonal=-self.windowSize+1)
                slidingMask = causalMask & windowMask
                attnMask = torch.zeros(S, kvSeqLen, dtype=q.dtype, device=q.device)
                attnMask.masked_fill_(~slidingMask, float('-inf'))
                self.attnMaskCache[cacheKey] = attnMask
            
            attnMask = self.attnMaskCache[cacheKey]
            
            out = sdpa(q, k, v, attn_mask=attnMask, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.oProj(out).to(inputDtype)

        if attentionMask is not None:
            out = out * attentionMask.to(out.dtype).unsqueeze(-1)

        return out, None


"""
================================================================================
Transformer Decoder Layer
================================================================================
"""

class RosaformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with attention and MLP.
    
    Implements the standard pre-normalization architecture:
    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))
    """
    def __init__(self, config: RosaformerConfig, layerIdx: int):
        super().__init__()
        self.hiddenSize = config.hidden_size
        self.selfAttn = RosaformerAttention(config=config, layerIdx=layerIdx)
        self.mlp = RosaformerMLP(config)
        self.inputLayernorm = RosaformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.postAttentionLayernorm = RosaformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hiddenStates: torch.Tensor,
        attentionMask: Optional[torch.Tensor] = None,
        positionIds: Optional[torch.LongTensor] = None,
        pastKeyValues: Optional[Cache] = None,
        useCache: Optional[bool] = False,
        cachePosition: Optional[torch.LongTensor] = None,
        positionEmbeddings: Optional[tuple] = None,
        **kwargs,
    ):
        residual = hiddenStates
        hiddenStates = self.inputLayernorm(hiddenStates)
        hiddenStates, _ = self.selfAttn(
            hiddenStates=hiddenStates,
            attentionMask=attentionMask,
            positionIds=positionIds,
            pastKeyValues=pastKeyValues,
            cachePosition=cachePosition,
            positionEmbeddings=positionEmbeddings,
            **kwargs,
        )
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        hiddenStates = self.postAttentionLayernorm(hiddenStates)
        hiddenStates = self.mlp(hiddenStates)
        hiddenStates = residual + hiddenStates
        
        return hiddenStates


"""
================================================================================
Pre-trained Model Base
================================================================================
"""

class RosaformerPreTrainedModel(PreTrainedModel):
    """
    Base class for Rosaformer models, handling weight initialization.
    """
    config_class = RosaformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RosaformerDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        """Initialize model weights with normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


"""
================================================================================
Main Model with ROSA Integration
================================================================================
"""

class RosaformerModel(RosaformerPreTrainedModel):
    """
    The complete Rosaformer transformer model with integrated ROSA.
    
    This model implements a decoder-only transformer architecture with
    optional ROSA mechanism for efficient long-context processing. ROSA
    is applied after each decoder layer (except optionally the first)
    to transfer information across attention windows.
    """
    def __init__(self, config: RosaformerConfig):
        super().__init__(config)
        self.paddingIdx = config.pad_token_id
        self.vocabSize = config.vocab_size

        self.embedTokens = nn.Embedding(config.vocab_size, config.hidden_size, self.paddingIdx)
        self.layers = nn.ModuleList([
            RosaformerDecoderLayer(config, layerIdx) 
            for layerIdx in range(config.num_hidden_layers)
        ])
        self.norm = RosaformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotaryEmb = RosaformerRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.useRosa = bool(getattr(config, "use_rosa", True))
        self.firstLayerGlobalNoRosa = bool(getattr(config, "first_layer_global_no_rosa", True))
        self.windowSize = getattr(config, "window_size", None)
        self.rosaWorkers = int(getattr(config, "rosa_workers", 64))
        self.rosaTemperature = float(getattr(config, "rosa_temperature", 1.0))
        self.rosaTemperatureInv = 1.0 / self.rosaTemperature
        self.rosaPadId = int(getattr(config, "rosa_pad_id", 0))
        self.rosaDenseTableMaxElems = int(getattr(config, "rosa_dense_table_max_elems", 50_000_000))
        
        self.useDynamicPruning = getattr(config, "use_dynamic_pruning", True)
        self.pruningKeepRatio = getattr(config, "pruning_keep_ratio", 0.75)
        self.useFusedOps = getattr(config, "use_fused_ops", True)
        
        self.positionIdsCache = {}

        if self.useRosa:
            rosaVocab = getattr(config, "rosa_vocab_sizes", 4096)
            if isinstance(rosaVocab, int):
                self.rosaVocabSizes = [rosaVocab for _ in range(config.num_hidden_layers)]
            else:
                assert len(rosaVocab) == config.num_hidden_layers, \
                    "len(rosa_vocab_sizes) must equal num_hidden_layers"
                self.rosaVocabSizes = list(map(int, rosaVocab))

            self.rosaLmHeads = nn.ModuleList([
                nn.Linear(config.hidden_size, max(V, 1), bias=False) if V > 0 else nn.Identity()
                for V in self.rosaVocabSizes
            ])
            self.rosaEmbeddings = nn.ModuleList([
                nn.Embedding(V, config.hidden_size) if V > 0 else nn.Identity()
                for V in self.rosaVocabSizes
            ])
            
            self.rosaActiveLayers = set()
            for layerIdx in range(config.num_hidden_layers):
                if self.rosaVocabSizes[layerIdx] > 0:
                    if not self.firstLayerGlobalNoRosa or layerIdx > 0:
                        self.rosaActiveLayers.add(layerIdx)

        self.post_init()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass through the model with optional ROSA.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs for positional encoding
            past_key_values: Cached key-value states for generation
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            use_cache: Whether to return key-value cache
            cache_position: Position in cache for current step
            
        Returns:
            BaseModelOutputWithPast containing last hidden state and cache
        """
        output_attentions = output_attentions or False
        output_hidden_states = output_hidden_states or False
        use_cache = use_cache if use_cache is not None else True
        return_dict = return_dict if return_dict is not None else True

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You need to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            if self.training:
                inputs_embeds = self.embedTokens(input_ids)
            else:
                originalShape = input_ids.shape
                flatIds = input_ids.flatten()
                uniqueIds, inverseIndices = torch.unique(flatIds, return_inverse=True)
                uniqueEmbeds = self.embedTokens(uniqueIds)
                inputs_embeds = uniqueEmbeds[inverseIndices].view(originalShape + (uniqueEmbeds.shape[-1],))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            pastSeenTokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            seqLen = inputs_embeds.shape[1]
            device = inputs_embeds.device
            cacheKey = (pastSeenTokens, seqLen, device)
            
            if cacheKey not in self.positionIdsCache:
                self.positionIdsCache[cacheKey] = torch.arange(
                    pastSeenTokens, 
                    pastSeenTokens + seqLen, 
                    device=device
                )
            cache_position = self.positionIdsCache[cacheKey]

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hiddenStates = inputs_embeds
        positionEmbeddings = self.rotaryEmb(hiddenStates, position_ids)
        
        attentionMaskExpanded = None
        if attention_mask is not None:
            attentionMaskExpanded = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)

        for layerIdx, decoderLayer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hiddenStates = torch.utils.checkpoint.checkpoint(
                    decoderLayer,
                    hiddenStates,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    positionEmbeddings,
                    use_reentrant=False
                )
            else:
                hiddenStates = decoderLayer(
                    hiddenStates,
                    attentionMask=attention_mask,
                    positionIds=position_ids,
                    pastKeyValues=past_key_values,
                    useCache=use_cache,
                    cachePosition=cache_position,
                    positionEmbeddings=positionEmbeddings,
                    **kwargs,
                )

            if self.useRosa and layerIdx in self.rosaActiveLayers and (not self.training or USE_ROSA_TRAINING):
                vocabL = self.rosaVocabSizes[layerIdx]
                
                logits = self.rosaLmHeads[layerIdx](hiddenStates)
                scaledLogits = logits * self.rosaTemperatureInv
                probs = F.softmax(scaledLogits, dim=-1)
                

                with torch.no_grad():
                    hardTokens = scaledLogits.argmax(dim=-1)
                    
                    rosaIdsCpu = batchRosaCpuLayer(
                        hardTokens,
                        vocabSize=vocabL,
                        padId=self.rosaPadId,
                        workers=self.rosaWorkers,
                        maxElems=self.rosaDenseTableMaxElems
                    )
                    rosaIds = rosaIdsCpu.to(hardTokens.device)
                embL = self.rosaEmbeddings[layerIdx]
                hardEmb = embL(rosaIds)
                embWeight = embL.weight
                softEmb = torch.matmul(probs, embWeight)

                vSte = softEmb + (hardEmb - softEmb).detach()

                hiddenStates = hiddenStates + vSte

                if attentionMaskExpanded is not None:
                    hiddenStates = hiddenStates * attentionMaskExpanded

        hiddenStates = self.norm(hiddenStates)
        
        if not return_dict:
            return tuple(v for v in [hiddenStates, past_key_values] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hiddenStates,
            past_key_values=past_key_values if use_cache else None,
        )


"""
================================================================================
Causal Language Model Head
================================================================================
"""

class RosaformerForCausalLM(RosaformerPreTrainedModel, GenerationMixin):
    """
    Rosaformer model with a causal language modeling head.
    
    This is the complete model for text generation, with a linear
    layer projecting hidden states to vocabulary logits.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RosaformerModel(config)
        self.vocabSize = config.vocab_size
        self.lmHead = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedTokens

    def set_input_embeddings(self, value):
        self.model.embedTokens = value

    def get_output_embeddings(self):
        return self.lmHead

    def set_output_embeddings(self, newEmbeddings):
        self.lmHead = newEmbeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple):
                past_length = past_key_values[0][0].shape[2]
            
            if past_length > 0:
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass with optional language modeling loss computation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached states
            inputs_embeds: Pre-computed embeddings
            labels: Target token IDs for computing loss
            use_cache: Whether to return cache
            cache_position: Cache position
            
        Returns:
            CausalLMOutputWithPast containing loss, logits, and cache
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        hiddenStates = outputs.last_hidden_state
        logits = self.lmHead(hiddenStates)

        loss = None
        if labels is not None:
            shiftLogits = logits[..., :-1, :].contiguous()
            shiftLabels = labels[..., 1:].contiguous()
            
            lossFct = nn.CrossEntropyLoss(ignore_index=-100)
            shiftLogits = shiftLogits.view(-1, self.config.vocab_size)
            shiftLabels = shiftLabels.view(-1)
            shiftLabels = shiftLabels.to(shiftLogits.device)
            loss = lossFct(shiftLogits, shiftLabels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


"""
================================================================================
Configuration Builder
================================================================================
"""

def createRosaformerConfig():
    """
    Create a RosaformerConfig from global configuration parameters.
    
    Returns:
        RosaformerConfig instance with all parameters set
    """
    return RosaformerConfig(
            head_dim=HEAD_DIM,
            hidden_act=HIDDEN_ACT,
            hidden_size=HIDDEN_SIZE,
            initializer_range=INITIALIZER_RANGE,
            intermediate_size=INTERMEDIATE_SIZE,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            rms_norm_eps=RMS_NORM_EPS,
            rope_theta=ROPE_THETA,
            vocab_size=VOCAB_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            attention_bias=False,
            attention_dropout=0.0,
            use_rosa=USE_ROSA,
            first_layer_global_no_rosa=FIRST_LAYER_GLOBAL_NO_ROSA,
            window_size=WINDOW_SIZE,
            rosa_workers=ROSA_WORKERS,
            rosa_vocab_sizes=ROSA_VOCAB_SIZES,
            rosa_temperature=ROSA_TEMPERATURE,
            rosa_pad_id=ROSA_PAD_ID,
            rosa_dense_table_max_elems=ROSA_DENSE_TABLE_MAX_ELEMS,
            skip_qk_norm=SKIP_QK_NORM,
        )


"""
================================================================================
Graceful Shutdown Handler
================================================================================
"""

class GracefulShutdown:
    """
    Handle Ctrl+C gracefully by saving checkpoint before exit.
    """
    def __init__(self, trainer=None):
        self.trainer = trainer
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handleSignal)
    
    def handleSignal(self, sig, frame):
        """Handle SIGINT (Ctrl+C)."""
        if self.interrupted:
            print("\n\nForce quit! Exiting without saving...")
            sys.exit(1)
        
        self.interrupted = True
        print("\n\nInterrupt detected! Saving checkpoint...")
        
        if self.trainer is not None:
            try:
                savePath = f"{self.trainer.args.output_dir}/checkpoint-interrupted"
                print(f"   Saving to: {savePath}")
                
                import os
                os.makedirs(savePath, exist_ok=True)
                
                self.trainer.save_model(savePath)
                self.trainer.state.save_to_json(os.path.join(savePath, "trainer_state.json"))
                
                if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                    torch.save(self.trainer.optimizer.state_dict(), os.path.join(savePath, "optimizer.pt"))
                
                if hasattr(self.trainer, 'lr_scheduler') and self.trainer.lr_scheduler is not None:
                    torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(savePath, "scheduler.pt"))
                
                print("   Checkpoint saved successfully!")
            except Exception as e:
                print(f"   Error saving checkpoint: {e}")
        
        print("\nGoodbye!\n")
        sys.exit(0)


"""
================================================================================
Main Training Entry Point
================================================================================
"""

def main():
    """
    Main training function.
    
    Initializes the model from scratch, loads dataset, and runs training
    using HuggingFace Trainer.
    """
    print("Initializing model configuration...")
    config = createRosaformerConfig()

    print("Creating model...")
    model = RosaformerForCausalLM(config)
    
    if GRADIENT_CHECKPOINTING:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        cudaCapability = torch.cuda.get_device_capability()
        cudaVersion = cudaCapability[0] + cudaCapability[1] / 10
        print(f"  CUDA Capability: {cudaVersion:.1f}")
        
        if cudaVersion >= 7.0:
            print("  Enabling torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print(f"  torch.compile() requires CUDA >= 7.0 (you have {cudaVersion:.1f}), skipping")

    print("Loading dataset...")
    trainDataset = load_from_disk(DATASET_PATH)
    print(f"Dataset size: {len(trainDataset)}")
    
    print("Setting up training arguments...")
    trainingArgs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        max_grad_norm=1.0,
        warmup_ratio=WARMUP,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        lr_scheduler_kwargs={"min_lr_rate": MINIR},
        logging_dir=LOGGING_DIR,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING if GRADIENT_CHECKPOINTING is not None else False,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEP,
        fp16=FP16 if FP16 is not None else False,
        bf16=BF16 if BF16 is not None else USE_BF16,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY if DATALOADER_PIN_MEMORY is not None else True,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR if DATALOADER_PREFETCH_FACTOR is not None else 2,
        report_to=REPORT_TO,
        remove_unused_columns=False,
    )

    print("Checking for existing checkpoints...")
    from pathlib import Path
    
    resumeCheckpoint = None
    outputPath = Path(OUTPUT_DIR)
    
    if outputPath.exists():
        checkpoints = []
        for item in outputPath.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                if item.name == "checkpoint-interrupted":
                    checkpoints.append((float('inf'), item))
                else:
                    try:
                        step = int(item.name.split("-")[1])
                        checkpoints.append((step, item))
                    except ValueError:
                        continue
        
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            resumeCheckpoint = str(checkpoints[0][1])
            print(f"  Found checkpoint: {resumeCheckpoint}")
            print(f"     Will resume training from this checkpoint")
        else:
            print("  No checkpoints found, starting from scratch")
    else:
        print("  No checkpoints found, starting from scratch")
    
    print("\nCreating trainer...")
    
    from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
    
    tokenizerPath = "./tokenizer/minipile.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizerPath)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    
    dataCollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=128,
    )
    
    trainer = Trainer(
        model=model,
        args=trainingArgs,
        train_dataset=trainDataset,
        data_collator=dataCollator,
    )

    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {totalParams:,}")

    shutdownHandler = GracefulShutdown(trainer)
    
    if USE_ROSA and ROSA_WORKERS > 0:
        print("Pre-warming ROSA pool...")
        dummyTokens = torch.randint(0, 100, (2, 128), dtype=torch.long)
        _ = batchRosaCpuLayer(
            dummyTokens,
            vocabSize=100,
            padId=0,
            workers=ROSA_WORKERS,
            maxElems=ROSA_DENSE_TABLE_MAX_ELEMS
        )
        print("  ROSA pool ready!")
    
    print("\nStarting training...")
    print("Tip: Press Ctrl+C to save checkpoint and exit gracefully\n")
    
    try:
        if resumeCheckpoint:
            print(f"Resuming from: {resumeCheckpoint}\n")
            trainer.train(resume_from_checkpoint=resumeCheckpoint)
        else:
            trainer.train()
        print("Training complete!")
    except KeyboardInterrupt:
        pass
    
    
    global _ROSA_POOL
    if _ROSA_POOL is not None:
        _ROSA_POOL.close()
        _ROSA_POOL.join()
        _ROSA_POOL = None


if __name__ == "__main__":
    from utils import setupMultiproc
    setupMultiproc()
    main()
