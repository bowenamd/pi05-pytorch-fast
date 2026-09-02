"""HIP int4_gemm + torch.library custom op (Inductor / torch.compile)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

_mod = None
_op_registered = False


def load_int4_gemm():
    global _mod
    if _mod is not None:
        return _mod

    from torch.utils.cpp_extension import load

    os.environ.setdefault("PYTORCH_ROCM_ARCH", os.environ.get("GPU_ARCHS", "gfx1151"))
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    os.environ.setdefault("HIP_PATH", "/opt/rocm")
    csrc = Path(__file__).resolve().parent / "csrc"
    import _rocm_sdk_core

    rocm_lib = Path(_rocm_sdk_core.__file__).resolve().parent / "lib"
    link_dir = Path("/tmp/pi05_fast_w4a4_rocm_lib")
    link_dir.mkdir(parents=True, exist_ok=True)
    soname = rocm_lib / "libamdhip64.so.7"
    link = link_dir / "libamdhip64.so"
    if soname.is_file() and not link.exists():
        link.symlink_to(soname)
    _mod = load(
        name="pi05_fast_w4a4_int4",
        sources=[str(csrc / "int4_gemm.cu")],
        extra_cuda_cflags=["-O3", "-std=c++20", "--rocm-path=/opt/rocm"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=[f"-L{link_dir}", f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"],
        verbose=os.environ.get("PI05_W4A4_VERBOSE", "0") == "1",
    )
    return _mod


def _register_custom_op() -> None:
    global _op_registered
    if _op_registered:
        return
    try:
        torch.ops.pi05_w4a4.int4_gemm3
        _op_registered = True
        return
    except (AttributeError, RuntimeError):
        pass

    def _fake_out(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch.empty(*x.shape[:-1], packed.shape[0], device=x.device, dtype=torch.float16)

    @torch.library.custom_op("pi05_w4a4::int4_gemm", mutates_args=())
    def _int4_gemm(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return load_int4_gemm().int4_gemm(x, packed)

    @_int4_gemm.register_fake
    def _int4_gemm_fake(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return _fake_out(x, packed)

    @torch.library.custom_op("pi05_w4a4::int4_gemm2", mutates_args=())
    def _int4_gemm2(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return load_int4_gemm().int4_gemm2(x, w0, w1)

    @_int4_gemm2.register_fake
    def _int4_gemm2_fake(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor):
        return _fake_out(x, w0), _fake_out(x, w1)

    @torch.library.custom_op("pi05_w4a4::int4_gemm3", mutates_args=())
    def _int4_gemm3(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return load_int4_gemm().int4_gemm3(x, w0, w1, w2)

    @_int4_gemm3.register_fake
    def _int4_gemm3_fake(
        x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ):
        return _fake_out(x, w0), _fake_out(x, w1), _fake_out(x, w2)

    _op_registered = True


_register_custom_op()


def int4_gemm(x: torch.Tensor, packed_w: torch.Tensor) -> torch.Tensor:
    """Compiled-graph-friendly INT4 GEMM. x: fp16 [..., K], packed: int32 [N, K/8+1]."""
    return torch.ops.pi05_w4a4.int4_gemm(x, packed_w)


def int4_gemm2(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.pi05_w4a4.int4_gemm2(x, w0, w1)


def int4_gemm3(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.pi05_w4a4.int4_gemm3(x, w0, w1, w2)


def preload() -> None:
    """JIT the HIP extension before torch.compile traces the graph."""
    load_int4_gemm()
    _register_custom_op()
