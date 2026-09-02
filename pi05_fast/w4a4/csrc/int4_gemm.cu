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
  TORCH_CHECK(w.size(1) == K / 8 + 1, name, " second dim must be K/8+1");
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

  if (K == 16384)
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
  dim3 grid((N + BN - 1) / BN, (static_cast<int>(qx.M) + BM - 1) / BM);
  gemm_iu4<<<grid, TPB, 0, stream>>>(
      qx.q.data_ptr<int>(),
      w.data_ptr<int>(),
      reinterpret_cast<_Float16*>(y.data_ptr<at::Half>()),
      qx.sa.data_ptr<float>(),
      static_cast<int>(qx.M),
      N,
      qx.K);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_gemm", &int4_gemm, "W4A4 packed INT4 GEMM (Hadamard on K in {1024,2048,16384})");
  m.def("int4_gemm2", &int4_gemm2, "Shared activation quant + two INT4 GEMMs");
  m.def("int4_gemm3", &int4_gemm3, "Shared activation quant + three INT4 GEMMs");
}
