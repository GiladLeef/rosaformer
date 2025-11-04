/*
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
*/

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using i32 = int32_t;
using i64 = int64_t;

static inline i64 off3(i64 b, i64 t, i64 m, i64 T, i64 M) {
    return ((b * T) + t) * M + m;
}

static inline i64 off4_bmsk(i64 b, i64 m, i64 s, i64 k, i64 M, i64 S, i64 K) {
    return (((b * M + m) * S) + s) * K + k;
}

static inline i64 off3_bms(i64 b, i64 m, i64 s, i64 M, i64 S) {
    return ((b * M + m) * S) + s;
}

static inline i64 off3_bmt(i64 b, i64 m, i64 t, i64 M, i64 T) {
    return ((b * M + m) * T) + t;
}

static inline i64 off2_bm(i64 b, i64 m, i64 M) {
    return b * M + m;
}

static inline i32 sam_new_state(
    i32* next_arr, i32* link, i32* length, i32* e,
    i32& size, const i32 K, const i32 L, const i64 row_strideK)
{
    i32 s = size;
    size += 1;
    length[s] = L;
    link[s]   = -1;
    e[s]      = -1;
    i32* row = next_arr + (i64)s * row_strideK;
    for (i32 j = 0; j < K; ++j) row[j] = -1;
    return s;
}

static inline void sam_extend(
    i32* next_arr, i32* link, i32* length, i32* e,
    i32& last_state, i32& size, const i32 K, const i32 x, const i32 pos,
    const i64 row_strideK)
{
    const i32 last = last_state;
    const i32 cur = sam_new_state(next_arr, link, length, e, size, K, length[last] + 1, row_strideK);
    i32 p = last;

    while (p != -1) {
        i32* prow = next_arr + (i64)p * row_strideK;
        if (prow[x] == -1) {
            prow[x] = cur;
            p = link[p];
        } else {
            break;
        }
    }

    if (p == -1) {
        link[cur] = 0;
    } else {
        i32* prow = next_arr + (i64)p * row_strideK;
        const i32 q = prow[x];
        if (length[p] + 1 == length[q]) {
            link[cur] = q;
        } else {
            const i32 clone = sam_new_state(next_arr, link, length, e, size, K, length[p] + 1, row_strideK);
            i32* crow = next_arr + (i64)clone * row_strideK;
            i32* qrow = next_arr + (i64)q     * row_strideK;
            for (i32 j = 0; j < K; ++j) crow[j] = qrow[j];
            link[clone] = link[q];
            e[clone]    = e[q];
            while (p != -1) {
                i32* prow2 = next_arr + (i64)p * row_strideK;
                if (prow2[x] == q) {
                    prow2[x] = clone;
                    p = link[p];
                } else break;
            }
            link[q]   = clone;
            link[cur] = clone;
        }
    }

    i32 v = cur;
    while (v != -1 && e[v] != pos) {
        e[v] = pos;
        v = link[v];
    }
    last_state = cur;
}

static inline i32 sam_match_next(const i32* next_arr, const i32* link,
                                 i32 last_state, const i32 x, const i32 row_strideK)
{
    i32 p = last_state;
    while (p != -1) {
        const i32* prow = next_arr + (i64)p * row_strideK;
        if (prow[x] != -1) return prow[x];
        p = link[p];
    }
    return -1;
}

at::Tensor rosa_batch_cpu_optimized(at::Tensor z_bt, i64 vocabSize, i64 padId) {
    TORCH_CHECK(z_bt.device().is_cpu(), "z_bt must be on CPU");
    TORCH_CHECK(z_bt.dtype() == at::kLong, "z_bt must be int64");
    TORCH_CHECK(z_bt.dim() == 2, "z_bt must be [B,T]");

    const i64 B = z_bt.size(0);
    const i64 T = z_bt.size(1);
    const i32 K = static_cast<i32>(vocabSize);
    const i64 S = 2 * T + 5;

    auto opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true);
    at::Tensor y_bt = at::full({B, T}, padId, opts);

    const int64_t* z_ptr = z_bt.data_ptr<int64_t>();
    int64_t* y_ptr = y_bt.data_ptr<int64_t>();

    #pragma omp parallel for schedule(static)
    for (i64 b = 0; b < B; ++b) {
        std::vector<i64> b_table(S * K, -1);
        std::vector<i64> c_table(S, -1);
        std::vector<i64> d_table(S, 0);
        std::vector<i64> e_table(S, -1);
        
        const int64_t* x = z_ptr + b * T;
        int64_t* y = y_ptr + b * T;
        
        i64 g = 0;
        i64 z = 1;
        
        for (i64 i = 0; i < T; ++i) {
            i64 t = x[i];
            i64 r = z;
            z += 1;
            d_table[r] = d_table[g] + 1;
            i64 p = g;
            
            while (p != -1 && b_table[p * K + t] == -1) {
                b_table[p * K + t] = r;
                p = c_table[p];
            }
            
            if (p == -1) {
                c_table[r] = 0;
            } else {
                i64 q = b_table[p * K + t];
                if (d_table[p] + 1 == d_table[q]) {
                    c_table[r] = q;
                } else {
                    i64 u = z;
                    z += 1;
                    for (i32 j = 0; j < K; ++j) {
                        b_table[u * K + j] = b_table[q * K + j];
                    }
                    d_table[u] = d_table[p] + 1;
                    c_table[u] = c_table[q];
                    e_table[u] = e_table[q];
                    
                    while (p != -1 && b_table[p * K + t] == q) {
                        b_table[p * K + t] = u;
                        p = c_table[p];
                    }
                    
                    c_table[q] = u;
                    c_table[r] = u;
                }
            }
            
            i64 vG = r;
            i64 a = padId;
            while (vG != -1) {
                if (d_table[vG] > 0 && e_table[vG] >= 0) {
                    i64 nxt = e_table[vG] + 1;
                    if (nxt < T) {
                        a = x[nxt];
                    }
                    break;
                }
                vG = c_table[vG];
            }
            y[i] = a;
            
            vG = g;
            while (vG != -1 && e_table[vG] < i) {
                e_table[vG] = i;
                vG = c_table[vG];
            }
            g = r;
        }
    }

    return y_bt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rosa_batch_cpu_optimized", &rosa_batch_cpu_optimized,
          "Optimized CPU ROSA batch processing",
          pybind11::arg("z_bt"),
          pybind11::arg("vocabSize"),
          pybind11::arg("padId"));
}

