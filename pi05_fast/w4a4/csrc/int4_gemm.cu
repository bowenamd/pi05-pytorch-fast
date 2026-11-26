// Embedl-compatible W4A4 GEMM: Hadamard (selected K) + packed INT4 + iu4 WMMA.
// Layout: X[M,K] fp16, W[N, K/8+1] int32 (last col = per-output-channel scale bits).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

typedef int int2v __attribute__((ext_vector_type(2)));
typedef int int8v __attribute__((ext_vector_type(8)));

#define BM 128
#define BN 128
#define BK 64
#define TPB 256
#define MI 2
#define NI 4
#define KPW (BK / 8)
#define LDAi (KPW + 1)
#define NLA ((BM * KPW) / TPB)
#define NLB ((BN * KPW) / TPB)

__global__ void gemm_iu4(const int* A, const int* B, _Float16* Cf, const float* sa, int M, int N, int K) {
  int ROW = K / 8 + 1;
  __shared__ int As[2][BM * LDAi];
  __shared__ int Bs[2][BN * LDAi];
  int bm = blockIdx.y * BM, bn = blockIdx.x * BN, tid = threadIdx.x;
  int wid = tid / 32, lid = tid % 32, lane = lid % 16, wm = wid / 2, wn = wid % 2;
  int8v c[MI][NI];
  for (int i = 0; i < MI; i++)
    for (int j = 0; j < NI; j++) c[i][j] = int8v{};
  for (int i = 0; i < NLA; i++) {
    int x = tid + i * TPB;
    int r = x / KPW, kk = x % KPW;
    int gr = bm + r;
    As[0][r * LDAi + kk] = (gr < M) ? A[gr * (K / 8) + kk] : 0;
  }
  for (int i = 0; i < NLB; i++) {
    int x = tid + i * TPB;
    int co = x / KPW, kk = x % KPW;
    int gc = bn + co;
    Bs[0][co * LDAi + kk] = (gc < N) ? B[gc * ROW + kk] : 0;
  }
  __syncthreads();
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    int nk = k0 + BK;
    int ra[NLA], rb[NLB];
    if (nk < K) {
      for (int i = 0; i < NLA; i++) {
        int x = tid + i * TPB;
        int r = x / KPW, kk = x % KPW;
        int gr = bm + r;
        ra[i] = (gr < M) ? A[gr * (K / 8) + (nk / 8 + kk)] : 0;
      }
      for (int i = 0; i < NLB; i++) {
        int x = tid + i * TPB;
        int co = x / KPW, kk = x % KPW;
        int gc = bn + co;
        rb[i] = (gc < N) ? B[gc * ROW + (nk / 8 + kk)] : 0;
      }
    }
    for (int ks = 0; ks < BK; ks += 16)
      for (int mi = 0; mi < MI; mi++)
        for (int ni = 0; ni < NI; ni++) {
          int r0 = wm * 32 + mi * 16, c0 = wn * 64 + ni * 16;
          int2v af, bf;
          af[0] = As[buf][(r0 + lane) * LDAi + ks / 8];
          af[1] = As[buf][(r0 + lane) * LDAi + ks / 8 + 1];
          bf[0] = Bs[buf][(c0 + lane) * LDAi + ks / 8];
          bf[1] = Bs[buf][(c0 + lane) * LDAi + ks / 8 + 1];
          c[mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af, true, bf, c[mi][ni], false);
        }
    __syncthreads();
    if (nk < K) {
      for (int i = 0; i < NLA; i++)
        As[buf ^ 1][(tid + i * TPB) / KPW * LDAi + (tid + i * TPB) % KPW] = ra[i];
      for (int i = 0; i < NLB; i++)
        Bs[buf ^ 1][(tid + i * TPB) / KPW * LDAi + (tid + i * TPB) % KPW] = rb[i];
      __syncthreads();
    }
    buf ^= 1;
  }
  for (int mi = 0; mi < MI; mi++)
    for (int ni = 0; ni < NI; ni++) {
      int r0 = wm * 32 + mi * 16, c0 = wn * 64 + ni * 16;
      int col = bn + c0 + lane;
      if (col >= N) continue;
      float scw = __int_as_float(B[col * ROW + (K / 8)]);
      for (int e = 0; e < 8; e++) {
        int r = 2 * e + lid / 16;
        int gr = bm + r0 + r;
        if (gr < M) Cf[gr * N + col] = (_Float16)((float)c[mi][ni][e] * sa[gr] * scw);
      }
    }
}

// v2: weight rows padded to K/8+4 words (scale still at word K/8) -> 128-bit loads,
// one barrier per K tile, LDS-staged coalesced epilogue. Bit-exact with gemm_iu4.
typedef int int4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float gelu_tanh(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + tanhf(inner));
}

// GLU=true: B = gate, B2 = up (same shape); C = fp16(gelu_tanh(fp16 gate) * fp16 up).
template <int TBM, int TBN, int TBK, int WGM, int WGN, bool GLU>
__global__ __launch_bounds__(WGM * WGN * 32) void gemm_iu4_v2(
    const int* __restrict__ A, const int* __restrict__ B, const int* __restrict__ B2,
    _Float16* __restrict__ C, const float* __restrict__ sa, int M, int N, int K, int LDB) {
  constexpr int NT = WGM * WGN * 32;
  constexpr int KW = TBK / 8;
  constexpr int LD4 = KW / 4 + 1;  // LDS row pitch in int4v (+1 pad)
  constexpr int CPR = KW / 4;
  constexpr int WTM = TBM / WGM, WTN = TBN / WGN;
  constexpr int VMI = WTM / 16, VNI = WTN / 16;
  constexpr int NB = GLU ? 2 : 1;
  constexpr int LA = (TBM * CPR + NT - 1) / NT, LB = (NB * TBN * CPR + NT - 1) / NT;
  static_assert(KW % 4 == 0 && WTM % 16 == 0 && WTN % 32 == 0);
  __shared__ int4v As[2][TBM * LD4];
  __shared__ int4v Bs[2][NB * TBN * LD4];
  static_assert(sizeof(As) >= WGM * WGN * 16 * 32 * sizeof(_Float16));
  const int KWT = K / 8;
  const int bm = blockIdx.y * TBM, bn = blockIdx.x * TBN, tid = threadIdx.x;
  const int wid = tid / 32, lid = tid % 32, lane = lid % 16, wm = wid / WGN, wn = wid % WGN;
  int4v ra[LA], rb[LB];
  auto gload = [&](int kw0) {
#pragma unroll
    for (int i = 0; i < LA; i++) {
      const int x = tid + i * NT, r = x / CPR, c = x % CPR, gr = bm + r;
      ra[i] = (x < TBM * CPR && gr < M) ? *(const int4v*)(A + (unsigned)(gr * KWT + kw0 + c * 4)) : int4v{};
    }
#pragma unroll
    for (int i = 0; i < LB; i++) {
      const int x = tid + i * NT, r = x / CPR, c = x % CPR, gc = bn + r % TBN;
      const int* src = (GLU && r >= TBN) ? B2 : B;
      rb[i] = (x < NB * TBN * CPR && gc < N) ? *(const int4v*)(src + (unsigned)(gc * LDB + kw0 + c * 4)) : int4v{};
    }
  };
  auto sstore = [&](int b) {
#pragma unroll
    for (int i = 0; i < LA; i++) {
      const int x = tid + i * NT;
      if (x < TBM * CPR) As[b][(x / CPR) * LD4 + x % CPR] = ra[i];
    }
#pragma unroll
    for (int i = 0; i < LB; i++) {
      const int x = tid + i * NT;
      if (x < NB * TBN * CPR) Bs[b][(x / CPR) * LD4 + x % CPR] = rb[i];
    }
  };
  int8v acc[NB][VMI][VNI];
#pragma unroll
  for (int g = 0; g < NB; g++)
#pragma unroll
    for (int i = 0; i < VMI; i++)
#pragma unroll
      for (int j = 0; j < VNI; j++) acc[g][i][j] = int8v{};
  gload(0);
  sstore(0);
  __syncthreads();
  int buf = 0;
  for (int k0 = 0; k0 < KWT; k0 += KW) {
    const bool more = k0 + KW < KWT;
    if (more) gload(k0 + KW);
#pragma unroll
    for (int kc = 0; kc < CPR; kc++) {
      int4v af[VMI], bf[NB][VNI];
#pragma unroll
      for (int mi = 0; mi < VMI; mi++) af[mi] = As[buf][(wm * WTM + mi * 16 + lane) * LD4 + kc];
#pragma unroll
      for (int g = 0; g < NB; g++)
#pragma unroll
        for (int ni = 0; ni < VNI; ni++) bf[g][ni] = Bs[buf][(g * TBN + wn * WTN + ni * 16 + lane) * LD4 + kc];
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int g = 0; g < NB; g++)
#pragma unroll
          for (int mi = 0; mi < VMI; mi++)
#pragma unroll
            for (int ni = 0; ni < VNI; ni++) {
              const int2v a2 = {af[mi][2 * h], af[mi][2 * h + 1]};
              const int2v b2 = {bf[g][ni][2 * h], bf[g][ni][2 * h + 1]};
              acc[g][mi][ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a2, true, b2, acc[g][mi][ni], false);
            }
    }
    // Double buffer: buf^1 was last read before the previous barrier, so one barrier suffices.
    if (more) {
      sstore(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
  }
  __syncthreads();
  _Float16* st = reinterpret_cast<_Float16*>(&As[0][0]) + wid * 16 * 32;
#pragma unroll
  for (int mi = 0; mi < VMI; mi++) {
    const int r0 = wm * WTM + mi * 16;
#pragma unroll
    for (int np = 0; np < VNI; np += 2) {
#pragma unroll
      for (int t = 0; t < 2; t++) {
        const int col = bn + wn * WTN + (np + t) * 16 + lane;
        const float scw = col < N ? __int_as_float(B[col * LDB + KWT]) : 0.f;
        const float scu = (GLU && col < N) ? __int_as_float(B2[col * LDB + KWT]) : 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int rr = 2 * e + lid / 16, gr = bm + r0 + rr;
          const float sav = gr < M ? sa[gr] : 0.f;
          const _Float16 g16 = (_Float16)((float)acc[0][mi][np + t][e] * sav * scw);
          if constexpr (GLU) {
            const _Float16 u16 = (_Float16)((float)acc[NB - 1][mi][np + t][e] * sav * scu);
            st[rr * 32 + t * 16 + lane] = (_Float16)(gelu_tanh((float)g16) * (float)u16);
          } else {
            st[rr * 32 + t * 16 + lane] = g16;
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const int idx = lid + j * 32, rr = idx / 4, cc = (idx % 4) * 8;
        const int gr = bm + r0 + rr, gcol = bn + wn * WTN + np * 16 + cc;
        if (gr < M && gcol < N) *(int4v*)(C + gr * N + gcol) = *(const int4v*)(st + rr * 32 + cc);
      }
      __syncthreads();
    }
  }
}

template <int K, int R>
__global__ void quant_rot_rb(const _Float16* X, int* Q, float* sc, int M, long RS) {
  int row = blockIdx.x;
  if (row >= M) return;
  int tid = threadIdx.x;
  __shared__ _Float16 lds[K];
  __shared__ float red[TPB];
  float reg[R];
#pragma unroll
  for (int j = 0; j < R; j++) reg[j] = (float)X[row * RS + tid * R + j];
#pragma unroll
  for (int len = 1; len < R; len <<= 1)
    for (int i = 0; i < R; i += 2 * len)
      for (int j = i; j < i + len; j++) {
        float a = reg[j], b = reg[j + len];
        reg[j] = a + b;
        reg[j + len] = a - b;
      }
#pragma unroll
  for (int j = 0; j < R; j++) lds[tid * R + j] = (_Float16)reg[j];
  __syncthreads();
  for (int len = R; len < K; len <<= 1) {
    for (int idx = tid; idx < K / 2; idx += TPB) {
      int group = idx / len, off = idx % len;
      int p = group * 2 * len + off;
      float a = (float)lds[p], b = (float)lds[p + len];
      lds[p] = (_Float16)(a + b);
      lds[p + len] = (_Float16)(a - b);
    }
    __syncthreads();
  }
  float m = 0;
  for (int k = tid; k < K; k += TPB) {
    float v = fabsf((float)lds[k]);
    m = fmaxf(m, v);
  }
  red[tid] = m;
  __syncthreads();
  for (int s = TPB / 2; s; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float sdiv = red[0] / 7.f + 1e-12f;
  if (tid == 0) sc[row] = sdiv * rsqrtf((float)K);
  for (int p = tid; p < K / 8; p += TPB) {
    int pk = 0;
    for (int j = 0; j < 8; j++) {
      int q = (int)lrintf((float)lds[p * 8 + j] / sdiv);
      q = max(-7, min(7, q));
      pk |= (q & 0xF) << (j * 4);
    }
    Q[row * (K / 8) + p] = pk;
  }
}

typedef _Float16 half8v __attribute__((ext_vector_type(8)));
typedef int int2q __attribute__((ext_vector_type(2)));

// Bit-exact with quant_rot_rb<16384, 64>: same FWHT stage order and fp16 rounding points,
// but stages 6..13 run in registers between two LDS transposes instead of 8 LDS passes.
__global__ __launch_bounds__(256) void quant_rot_16k(const _Float16* __restrict__ X, int* __restrict__ Q,
                                                     float* __restrict__ sc, int M, long RS) {
  constexpr int K = 16384;
  const int row = blockIdx.x;
  if (row >= M) return;
  const int tid = threadIdx.x;
  __shared__ __attribute__((aligned(16))) _Float16 lds[K];
  __shared__ float red[256];
  float v[64];
  // phase 1: thread owns bits 6..13, registers hold bits 0..5 (fp32, like the R=64 register stages)
  const half8v* src = reinterpret_cast<const half8v*>(X + row * RS + tid * 64);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const half8v h = src[i];
#pragma unroll
    for (int j = 0; j < 8; j++) v[i * 8 + j] = (float)h[j];
  }
#pragma unroll
  for (int len = 1; len < 64; len <<= 1)
#pragma unroll
    for (int i = 0; i < 64; i += 2 * len)
#pragma unroll
      for (int j = i; j < i + len; j++) {
        const float a = v[j], b = v[j + len];
        v[j] = a + b;
        v[j + len] = a - b;
      }
  half8v* dst = reinterpret_cast<half8v*>(lds + tid * 64);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    half8v h;
#pragma unroll
    for (int j = 0; j < 8; j++) h[j] = (_Float16)v[i * 8 + j];
    dst[i] = h;
  }
  __syncthreads();
  // phase 2: thread owns bits 0..5 and 12..13, registers hold bits 6..11 (fp16 rounding per stage)
  const int base = ((tid >> 6) << 12) | (tid & 63);
#pragma unroll
  for (int j = 0; j < 64; j++) v[j] = (float)lds[base + (j << 6)];
#pragma unroll
  for (int len = 1; len < 64; len <<= 1)
#pragma unroll
    for (int i = 0; i < 64; i += 2 * len)
#pragma unroll
      for (int j = i; j < i + len; j++) {
        const float a = v[j], b = v[j + len];
        v[j] = (float)(_Float16)(a + b);
        v[j + len] = (float)(_Float16)(a - b);
      }
#pragma unroll
  for (int j = 0; j < 64; j++) lds[base + (j << 6)] = (_Float16)v[j];
  __syncthreads();
  // phase 3: thread owns bits 4..11, registers hold bits 0..3 (16 consecutive) x bits 12..13
#pragma unroll
  for (int g = 0; g < 4; g++)
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const half8v h = *reinterpret_cast<const half8v*>(lds + (g << 12) + (tid << 4) + i * 8);
#pragma unroll
      for (int j = 0; j < 8; j++) v[g * 16 + i * 8 + j] = (float)h[j];
    }
#pragma unroll
  for (int len = 16; len < 64; len <<= 1)
#pragma unroll
    for (int i = 0; i < 64; i += 2 * len)
#pragma unroll
      for (int j = i; j < i + len; j++) {
        const float a = v[j], b = v[j + len];
        v[j] = (float)(_Float16)(a + b);
        v[j + len] = (float)(_Float16)(a - b);
      }
  float m = 0;
#pragma unroll
  for (int j = 0; j < 64; j++) m = fmaxf(m, fabsf(v[j]));
  red[tid] = m;
  __syncthreads();
  for (int s = 128; s; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float sdiv = red[0] / 7.f + 1e-12f;
  if (tid == 0) sc[row] = sdiv * rsqrtf((float)K);
#pragma unroll
  for (int g = 0; g < 4; g++) {
    int2q w2;
#pragma unroll
    for (int w = 0; w < 2; w++) {
      int pk = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int q = (int)lrintf(v[g * 16 + w * 8 + j] / sdiv);
        q = max(-7, min(7, q));
        pk |= (q & 0xF) << (j * 4);
      }
      w2[w] = pk;
    }
    *reinterpret_cast<int2q*>(Q + row * (K / 8) + (g << 9) + (tid << 1)) = w2;
  }
}

__global__ void quant_rows(const _Float16* X, int* Q, float* sc, int M, int K) {
  int row = blockIdx.x;
  if (row >= M) return;
  __shared__ float sm[TPB];
  float m = 0;
  for (int k = threadIdx.x; k < K; k += TPB) {
    float v = fabsf((float)X[row * K + k]);
    m = fmaxf(m, v);
  }
  sm[threadIdx.x] = m;
  __syncthreads();
  for (int s = TPB / 2; s; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  float s0 = sm[0] / 7.f + 1e-12f;
  if (threadIdx.x == 0) sc[row] = s0;
  for (int p = threadIdx.x; p < K / 8; p += TPB) {
    int pk = 0;
    for (int j = 0; j < 8; j++) {
      int q = (int)lrintf((float)X[row * K + p * 8 + j] / s0);
      q = max(-7, min(7, q));
      pk |= (q & 0xF) << (j * 4);
    }
    Q[row * (K / 8) + p] = pk;
  }
}

struct QuantizedX {
  torch::Tensor x2;
  torch::Tensor q;
  torch::Tensor sa;
  long M = 0;
  int K = 0;
};

static void check_packed(const torch::Tensor& w, int K, const char* name) {
  TORCH_CHECK(w.is_cuda(), name, " must be on CUDA/HIP");
  TORCH_CHECK(w.scalar_type() == torch::kInt32 && w.dim() == 2, name, " must be int32 [N, K/8+1]");
  TORCH_CHECK(w.size(1) == K / 8 + 1 || w.size(1) == K / 8 + 4, name, " second dim must be K/8+1 (or K/8+4 padded)");
  TORCH_CHECK(w.stride(1) == 1, name, " must be row-contiguous");
}

static QuantizedX quantize_x(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "int4_gemm expects CUDA/HIP tensors");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "int4_gemm activations must be float16");
  TORCH_CHECK(x.size(-1) > 0);
  const int K = static_cast<int>(x.size(-1));
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64, got ", K);

  QuantizedX qx;
  qx.x2 = x.contiguous();
  qx.K = K;
  qx.M = 1;
  for (int i = 0; i + 1 < qx.x2.dim(); i++) qx.M *= qx.x2.size(i);

  const auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(qx.x2.device());
  const auto opts_f = torch::TensorOptions().dtype(torch::kFloat).device(qx.x2.device());
  qx.q = torch::empty({qx.M, (long)(K / 8)}, opts_i);
  qx.sa = torch::empty({qx.M}, opts_f);

  const c10::cuda::CUDAGuard guard(qx.x2.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const _Float16* xp = reinterpret_cast<const _Float16*>(qx.x2.data_ptr<at::Half>());
  int* qp = qx.q.data_ptr<int>();
  float* sap = qx.sa.data_ptr<float>();

  if (K == 16384 && reinterpret_cast<uintptr_t>(xp) % 16 == 0) {
    quant_rot_16k<<<static_cast<int>(qx.M), TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  } else if (K == 16384)
    quant_rot_rb<16384, 64><<<static_cast<int>(qx.M), TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else if (K == 2048)
    quant_rot_rb<2048, 8><<<static_cast<int>(qx.M), TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else if (K == 1024)
    quant_rot_rb<1024, 4><<<static_cast<int>(qx.M), TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);
  else
    quant_rows<<<static_cast<int>(qx.M), TPB, 0, stream>>>(xp, qp, sap, static_cast<int>(qx.M), K);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return qx;
}

static torch::Tensor gemm_from_q(const QuantizedX& qx, torch::Tensor w) {
  check_packed(w, qx.K, "packed W");
  const int N = static_cast<int>(w.size(0));
  const auto opts_h = torch::TensorOptions().dtype(torch::kHalf).device(qx.x2.device());
  auto y = torch::empty({qx.M, (long)N}, opts_h);

  const c10::cuda::CUDAGuard guard(qx.x2.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (w.size(1) == qx.K / 8 + 4) {
    TORCH_CHECK(N % 16 == 0, "padded INT4 weights need N % 16 == 0, got ", N);
    const int M = static_cast<int>(qx.M), K = qx.K, LDB = static_cast<int>(w.stride(0));
    const int* A = qx.q.data_ptr<int>();
    const int* B = w.data_ptr<int>();
    _Float16* C = reinterpret_cast<_Float16*>(y.data_ptr<at::Half>());
    const float* sa = qx.sa.data_ptr<float>();
    // Tile configs picked per shape on gfx1151 (M=640 microbench, scripts/gemm_micro.py).
    if (N <= 512 && K % 128 == 0) {
      dim3 grid((N + 127) / 128, (M + 63) / 64);
      gemm_iu4_v2<64, 128, 128, 2, 2, false><<<grid, 128, 0, stream>>>(A, B, nullptr, C, sa, M, N, K, LDB);
    } else if (K >= 8192 && K % 128 == 0) {
      dim3 grid((N + 127) / 128, (M + 127) / 128);
      gemm_iu4_v2<128, 128, 128, 4, 2, false><<<grid, 256, 0, stream>>>(A, B, nullptr, C, sa, M, N, K, LDB);
    } else {
      dim3 grid((N + 127) / 128, (M + 127) / 128);
      gemm_iu4_v2<128, 128, 64, 4, 2, false><<<grid, 256, 0, stream>>>(A, B, nullptr, C, sa, M, N, K, LDB);
    }
  } else {
    dim3 grid((N + BN - 1) / BN, (static_cast<int>(qx.M) + BM - 1) / BM);
    gemm_iu4<<<grid, TPB, 0, stream>>>(
        qx.q.data_ptr<int>(),
        w.data_ptr<int>(),
        reinterpret_cast<_Float16*>(y.data_ptr<at::Half>()),
        qx.sa.data_ptr<float>(),
        static_cast<int>(qx.M),
        N,
        qx.K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto yshape = qx.x2.sizes().vec();
  yshape.back() = N;
  return y.view(yshape);
}

static torch::Tensor int4_gemm(torch::Tensor x, torch::Tensor w) {
  return gemm_from_q(quantize_x(x), w);
}

static std::tuple<torch::Tensor, torch::Tensor> int4_gemm2(
    torch::Tensor x, torch::Tensor w0, torch::Tensor w1) {
  auto qx = quantize_x(x);
  return {gemm_from_q(qx, w0), gemm_from_q(qx, w1)};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> int4_gemm3(
    torch::Tensor x, torch::Tensor w0, torch::Tensor w1, torch::Tensor w2) {
  auto qx = quantize_x(x);
  return {gemm_from_q(qx, w0), gemm_from_q(qx, w1), gemm_from_q(qx, w2)};
}

static torch::Tensor int4_gemm_glu(torch::Tensor x, torch::Tensor wg, torch::Tensor wu) {
  auto qx = quantize_x(x);
  const int K = qx.K, M = static_cast<int>(qx.M), N = static_cast<int>(wg.size(0));
  check_packed(wg, K, "gate W");
  check_packed(wu, K, "up W");
  TORCH_CHECK(wg.size(1) == K / 8 + 4 && wg.sizes() == wu.sizes() && wg.stride(0) == wu.stride(0),
              "int4_gemm_glu needs padded (K/8+4) gate/up weights of identical layout");
  TORCH_CHECK(N % 64 == 0 && K % 64 == 0, "int4_gemm_glu needs N % 64 == 0");
  auto y = torch::empty({qx.M, (long)N}, torch::TensorOptions().dtype(torch::kHalf).device(qx.x2.device()));
  const c10::cuda::CUDAGuard guard(qx.x2.device());
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(N / 64, (M + 127) / 128);
  gemm_iu4_v2<128, 64, 64, 4, 2, true><<<grid, 256, 0, stream>>>(
      qx.q.data_ptr<int>(), wg.data_ptr<int>(), wu.data_ptr<int>(),
      reinterpret_cast<_Float16*>(y.data_ptr<at::Half>()), qx.sa.data_ptr<float>(),
      M, N, K, static_cast<int>(wg.stride(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto yshape = qx.x2.sizes().vec();
  yshape.back() = N;
  return y.view(yshape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_gemm", &int4_gemm, "W4A4 packed INT4 GEMM (Hadamard on K in {1024,2048,16384})");
  m.def("int4_gemm2", &int4_gemm2, "Shared activation quant + two INT4 GEMMs");
  m.def("int4_gemm_glu", &int4_gemm_glu, "Shared activation quant + gate/up INT4 GEMM fused with gelu_tanh(gate)*up");
  m.def("int4_gemm3", &int4_gemm3, "Shared activation quant + three INT4 GEMMs");
}
