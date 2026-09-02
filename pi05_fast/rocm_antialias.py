"""Dodge the gfx1151 L2 set-aliasing cliff from inside Inductor (no PyTorch fork).

Vendored from rocm-scripts/test/pytorch/_antialias.py (lerobot_pi05 / openpi_pi05).

On gfx1151, when a GEMM operand's leading dimension (in *bytes*) is a multiple of
the 2 KB L2 aliasing period, the ~128 rows a tile streams concurrently all index
into the same L2 set, conflict-miss, and serialize -- costing 20-25% on the
affected GEMMs (e.g. PaliGemma down_proj, K=16384). Nudging a leading dimension a
little off the period spreads the rows across sets; the logical shape and the
contracted K are unchanged -> no extra MACs.

`enable_antialias()` turns on three off-period stride adjustments, each
independently toggleable via env vars:

  1. Weight `lda`  -- pad + slice ops inserted into `freezing_passes`, which
     freezing's constant_fold collapses into a pre-padded frozen param at
     compile time (no runtime copy). Always on. It must run there and not in the
     post-grad pass: post-grad runs *after* Inductor computes the FX-graph cache
     key, so mutating a frozen param there lets a later cache hit pair the
     cached kernel's baked-in padded stride with an unpadded tensor, and the
     GEMM reads ~128 KiB out of bounds (AIESW-41923).
  2. GEMM output `ldd` -- off-period output row stride (ANTIALIAS_MM_OUTPUT, on).
  3. Activation `ldb`  -- runtime pad of the activation (ANTIALIAS_ACT_PAD, on).

Call enable_antialias() once before torch.compile.
"""

from __future__ import annotations

import logging
import os

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

log = logging.getLogger("antialias")

# Shared off-period geometry (gfx1151).
PERIOD_BYTES = 2048   # L2 aliasing period: bump leading dims that are a multiple of this
ALIGN_BYTES = 128     # keep the bumped stride 128 B-aligned (>= cache line)
MIN_OUTER = 16        # only outputs whose outer dim exceeds this


# =============================================================================
# Shared helper: off-period 2D output stride
# =============================================================================
def _offperiod_stride(size, dtype):
    """Return an off-period contiguous-ish 2D stride [N + align, 1], or None.

    None means "leave the default contiguous stride" (shape not 2D, dynamic,
    outer dim too small, or row pitch already off the aliasing period).
    """
    if len(size) != 2:
        return None
    try:
        m = int(size[0])
        n = int(size[1])
    except (TypeError, ValueError):
        return None  # symbolic / dynamic shapes
    it = dtype.itemsize
    align = ALIGN_BYTES // it
    period = PERIOD_BYTES // it
    if align and period and m > MIN_OUTER and n % period == 0:
        return [n + align, 1]
    return None


# =============================================================================
# GEMM-output (ldd) striding -- monkeypatches, installed on demand
# =============================================================================
# torch 2.12 derives the GEMM output layout from two sources that must agree, or
# MultiTemplateBuffer.finalize_as_triton_caller asserts buffer.get_stride() ==
# caller.layout.stride:
#   * mm_common.mm_args        -> MultiTemplateBuffer + extern (aten) choice
#   * MMKernelInputs.output_layout() -> Triton template choices (torch 2.12+)
# We apply the identical off-period transform in both so the buffer and the
# chosen Triton caller match and the off-period pitch is honoured on every
# backend. output_layout must return a FixedLayout (a Flexible one would
# recompute contiguous strides). No-op unless comprehensive_padding is on.
_OFFPERIOD_INSTALLED = False


def _install_offperiod_mm_output():
    """Idempotently patch mm_args and MMKernelInputs.output_layout."""
    global _OFFPERIOD_INSTALLED
    if _OFFPERIOD_INSTALLED:
        return
    _OFFPERIOD_INSTALLED = True

    import torch._inductor.kernel.mm_common as mmc
    from torch._inductor import config
    from torch._inductor.ir import FixedLayout

    # --- 1. mm_args: layout for the MultiTemplateBuffer and the extern choice ---
    _orig_mm_args = mmc.mm_args

    def _mm_args(*args, **kwargs):
        out = _orig_mm_args(*args, **kwargs)
        if not config.comprehensive_padding:
            return out
        m, n, k, layout, *rest = out
        try:
            if isinstance(layout, FixedLayout) and len(layout.size) == 2:
                n_cols = int(layout.size[1])
                s0, s1 = int(layout.stride[0]), int(layout.stride[1])
                if (s0, s1) == (n_cols, 1):  # only the default contiguous output
                    ns = _offperiod_stride(layout.size, layout.dtype)
                    if ns is not None:
                        layout = FixedLayout(layout.device, layout.dtype,
                                             list(layout.size), ns)
                        out = [m, n, k, layout, *rest]
        except (TypeError, ValueError):
            pass  # symbolic shapes / non-concrete strides: leave unchanged
        except Exception as e:  # never break compilation over this
            log.warning("offperiod mm_args skipped: %s", e)
        return out

    mmc.mm_args = _mm_args
    for _modname in ("mm", "bmm", "mm_plus_mm", "mm_grouped"):
        try:
            _mod = __import__(f"torch._inductor.kernel.{_modname}",
                              fromlist=["mm_args"])
            if hasattr(_mod, "mm_args"):
                _mod.mm_args = _mm_args
        except Exception:
            pass

    # --- 2. MMKernelInputs.output_layout: layout for the Triton template choices
    # (torch 2.12+; older releases route templates through mm_args and skip this.)
    output_layout_patched = False
    try:
        from torch._inductor.kernel_inputs import MMKernelInputs

        _orig_output_layout = MMKernelInputs.output_layout

        def _output_layout(self, flexible: bool = True):
            lay = _orig_output_layout(self, flexible=flexible)
            if not config.comprehensive_padding:
                return lay
            try:
                ns = _offperiod_stride(lay.size, lay.dtype)
                if ns is not None:
                    return FixedLayout(lay.device, lay.dtype, list(lay.size), ns)
            except (TypeError, ValueError):
                pass
            except Exception as e:
                log.warning("offperiod output_layout skipped: %s", e)
            return lay

        MMKernelInputs.output_layout = _output_layout
        output_layout_patched = True
    except Exception:
        pass  # pre-2.12: mm_args path is sufficient

    log.info("offperiod mm output installed (output_layout patched=%s)",
             output_layout_patched)

    # --- optional verifier: buffer/caller strides must agree at finalize time --
    if os.environ.get("ANTIALIAS_DEBUG_FINALIZE") == "1":
        try:
            from torch._inductor.ir import MultiTemplateBuffer as _MTB

            _orig_fin = _MTB.finalize_as_triton_caller

            def _fin(self, caller):
                bs = list(self.get_stride())
                cs = list(caller.layout.stride)
                tag = "ok" if bs == cs else "MISMATCH"
                print(f"[finalize {tag}] buffer={bs} caller={cs}", flush=True)
                return _orig_fin(self, caller)

            _MTB.finalize_as_triton_caller = _fin
            print("[antialias] finalize verifier active", flush=True)
        except Exception as e:
            print(f"[antialias] verifier setup failed: {e}", flush=True)


# =============================================================================
# Weight (lda) off-period padding at freezing time -- graph ops + constant fold
# =============================================================================
# This has to happen at freezing time, not in the post-grad pass. Post-grad runs
# *after* Inductor computes the FX-graph cache key, so rewriting a frozen weight
# in place there is invisible to the key: on a cache hit post-grad never re-runs,
# the module still holds the unpadded weight, and the cached kernel has the
# padded leading stride baked in as a literal -- the GEMM then reads ~128 KiB
# past the end of the allocation (AIESW-41923). The fault surfaces as
# HSA_STATUS_ERROR_MEMORY_FAULT and usually *hangs* the process rather than
# aborting it, so anything exercising this path needs a timeout.
#
# Expressing the pad as graph ops before freezing's constant_fold instead means
# the padded constant exists *before* the key is computed, so the key and the
# generated code can never disagree. Same approach upstream uses in
# convert_conv_weights_to_channels_last, and constant_fold hands the folded view
# straight to register_buffer (no clone), so the off-period stride survives.
aten = torch.ops.aten

_FREEZE_PAD_INSTALLED = False


def _install_freezing_weight_pad(period_bytes: int = PERIOD_BYTES,
                                 bump_bytes: int = ALIGN_BYTES) -> None:
    """Idempotently run the weight pad inside Inductor's freezing_passes."""
    global _FREEZE_PAD_INSTALLED
    if _FREEZE_PAD_INSTALLED:
        return
    _FREEZE_PAD_INSTALLED = True

    import torch._inductor.freezing as _fz

    _orig_freezing_passes = _fz.freezing_passes

    def _freezing_passes(gm, example_inputs, *args, **kwargs):
        _orig_freezing_passes(gm, example_inputs, *args, **kwargs)
        try:
            n = _pad_weights_via_graph_ops(gm, period_bytes, bump_bytes)
        except Exception as e:  # never break compilation over this
            log.warning("antialias: freezing weight pad skipped: %s", e)
            return
        if n:
            log.warning("antialias: padded %d weights via constant folding", n)

    _fz.freezing_passes = _freezing_passes
    log.info("offperiod freezing weight pad installed")


def _is_const_foldable(node, memo: dict) -> bool:
    """True if `node` evaluates to a compile-time constant, i.e. freezing's
    constant_fold will collapse it (and anything we append) into a get_attr."""
    if not isinstance(node, torch.fx.Node):
        return True
    cached = memo.get(node)
    if cached is not None:
        return cached
    memo[node] = False  # cycle guard
    if node.op == "get_attr":
        result = True
    elif node.op == "call_function":
        result = all(_is_const_foldable(a, memo) for a in node.all_input_nodes)
    else:
        result = False  # placeholder / output
    memo[node] = result
    return result


def _is_target_device(t) -> bool:
    """Only pad GPU operands -- the aliasing cliff being dodged is gfx1151's L2.

    Named rather than inlined so tests can exercise the graph transform without
    a live GPU context ("cuda" covers ROCm too under torch's HIP mapping).
    """
    return t.device.type == "cuda"


def _node_tensor(gm, node):
    """Fake or real tensor for a graph node, whichever is available."""
    val = node.meta.get("val")
    if val is not None:
        return val
    if node.op == "get_attr":
        return getattr(gm, node.target, None)
    return None


def _emit_offperiod_pad(graph, w_node, shape, axis: int, bump: int):
    """Insert pad+slice so `w_node`'s leading stride grows by `bump` elements.

    Logical shape and values are unchanged -- only the pitch moves off the
    aliasing period, so no extra MACs. constant_fold collapses the whole chain
    into a single pre-padded frozen param.

    Inserted directly after `w_node` so the result dominates every user, which
    lets a weight shared by several GEMMs (e.g. z_proj on z_init and z_goal) be
    padded once instead of once per consumer.
    """
    d0, d1 = int(shape[0]), int(shape[1])
    # inserting_before(w_node.next), not inserting_after(w_node): the latter
    # re-anchors on w_node for every node created, emitting the chain reversed.
    with graph.inserting_before(w_node.next):
        if axis == 0:
            # row-major [d0, d1] stride (d1, 1) -> (d1 + bump, 1)
            padded = graph.call_function(
                aten.constant_pad_nd.default, (w_node, [0, bump], 0.0))
            return graph.call_function(aten.slice.Tensor, (padded, 1, 0, d1))
        # col-major [d0, d1] stride (1, d0) -> (1, d0 + bump). The transpose is
        # the contiguous view, so pad that and transpose back.
        t = graph.call_function(aten.permute.default, (w_node, [1, 0]))
        padded = graph.call_function(
            aten.constant_pad_nd.default, (t, [0, bump], 0.0))
        sliced = graph.call_function(aten.slice.Tensor, (padded, 1, 0, d0))
        return graph.call_function(aten.permute.default, (sliced, [1, 0]))


def _pad_weights_via_graph_ops(gm, period_bytes: int, bump_bytes: int) -> int:
    """Bump every on-period constant GEMM weight off the aliasing period."""
    graph = gm.graph
    memo: dict = {}
    # A weight feeding several GEMMs (z_proj serves both z_init and z_goal)
    # should be padded once. Key on the get_attr target, since tracing can emit a
    # separate get_attr per access site for the same tensor. Reuse only when the
    # existing chain's anchor precedes this consumer, otherwise the replacement
    # would be a use before its definition. `order` is captured before any
    # mutation; anchors and GEMM nodes are all original nodes.
    order = {n: i for i, n in enumerate(graph.nodes)}
    done: dict = {}   # key -> (anchor w_node, padded replacement)
    padded = 0
    for op, (_, i2) in _MM_OPS.items():
        for node in graph.find_nodes(op="call_function", target=op):
            w_node = node.args[i2]
            if not isinstance(w_node, torch.fx.Node):
                continue
            key = w_node.target if w_node.op == "get_attr" else w_node
            prev = done.get(key)
            if prev is not None and order[prev[0]] < order[node]:
                node.replace_input_with(w_node, prev[1])
                continue
            if not _is_const_foldable(w_node, memo):
                continue  # runtime activation, not a weight
            w = _node_tensor(gm, w_node)
            if w is None or w.dim() != 2 or not _is_target_device(w):
                continue
            lead = _leading_stride_axis(w)
            if lead is None:
                continue
            axis, stride_elems = lead
            if not _on_period(stride_elems, w.element_size(), period_bytes):
                continue
            bump = bump_bytes // w.element_size()
            if bump <= 0:
                continue
            new_node = _emit_offperiod_pad(graph, w_node, w.shape, axis, bump)
            node.replace_input_with(w_node, new_node)
            done[key] = (w_node, new_node)
            padded += 1
    if padded:
        graph.lint()
        gm.recompile()
    return padded


# =============================================================================
# Activation (ldb) off-period padding -- Inductor post-grad pass
# =============================================================================

# mm-family ops and the (mat1, mat2) arg positions of their matrix operands.
_MM_OPS = {
    aten.mm.default: (0, 1),
    aten.addmm.default: (1, 2),
}


def _leading_stride_axis(t: torch.Tensor):
    """Return (axis, stride_elems) of the 2D operand's leading (non-unit) stride.

    A row/col-major 2D matrix has one unit stride (the contiguous axis) and one
    'leading' stride. That leading stride is what BLAS reports as lda/ldb.
    """
    if t.dim() != 2:
        return None
    s0, s1 = t.stride()
    if s0 == 1 and s1 > 1:
        return 1, s1
    if s1 == 1 and s0 > 1:
        return 0, s0
    return None  # not a plain 2D matrix view (both unit / both non-unit)


def _on_period(stride_elems: int, itemsize: int, period_bytes: int) -> bool:
    byte_stride = stride_elems * itemsize
    return byte_stride != 0 and byte_stride % period_bytes == 0


class AntiAliasStridePass(CustomGraphPass):
    """Bump on-period GEMM *activation* leading dims off the L2 aliasing period.

    Weights are handled earlier, by _install_freezing_weight_pad; a post-grad
    pass runs after the FX-graph cache key is computed and so cannot safely
    touch frozen constants.

    Parameters
    ----------
    period_bytes : aliasing period (gfx1151 = 2048).
    bump_bytes   : how far off the period to nudge (>= cache line; 128 measured best).
    pad_activation : bump the runtime activation operand (adds a runtime pad).
    """

    def __init__(
        self,
        period_bytes: int = PERIOD_BYTES,
        bump_bytes: int = ALIGN_BYTES,
        pad_activation: bool = False,
    ) -> None:
        self.period_bytes = period_bytes
        self.bump_bytes = bump_bytes
        self.pad_activation = pad_activation
        self.n_act_padded = 0

    # ---- CustomGraphPass API -------------------------------------------------
    def uuid(self):
        # Part of Inductor's FX-graph cache key: bump when behaviour changes.
        return ("antialias-stride", 3, self.period_bytes, self.bump_bytes,
                self.pad_activation)

    def __call__(self, graph: torch.fx.Graph) -> None:
        if not self.pad_activation:
            return
        for op, (i1, _) in _MM_OPS.items():
            for node in graph.find_nodes(op="call_function", target=op):
                self._maybe_pad_activation(graph, node, node.args[i1], i1)
        if self.n_act_padded:
            log.warning("AntiAliasStridePass: padded %d activations",
                        self.n_act_padded)

    # ---- activation (runtime, optional) -------------------------------------
    def _maybe_pad_activation(self, graph, node, a_node, arg_idx) -> None:
        if not isinstance(a_node, torch.fx.Node):
            return
        val = a_node.meta.get("val")
        if val is None or val.dim() != 2 or val.device.type != "cuda":
            return
        lead = _leading_stride_axis(val)
        if lead is None:
            return
        axis, stride_elems = lead
        if not _on_period(stride_elems, val.element_size(), self.period_bytes):
            return
        bump = self.bump_bytes // val.element_size()
        if bump <= 0 or axis != 0:
            return  # only handle the common row-major [M,K] activation

        K = val.shape[1]
        with graph.inserting_before(node):
            padded = graph.call_function(
                aten.constant_pad_nd.default, (a_node, [0, bump], 0.0)
            )
            sliced = graph.call_function(
                aten.slice.Tensor, (padded, 1, 0, K)
            )
        new_args = list(node.args)
        new_args[arg_idx] = sliced
        node.args = tuple(new_args)
        self.n_act_padded += 1


# =============================================================================
# Public entry point
# =============================================================================
def enable_antialias():
    """Enable the gfx1151 anti-alias stride dodge. Call once before torch.compile.

    Always applies freezing + the weight-lda pad. GEMM-output (ldd) striding is
    on by default (ANTIALIAS_MM_OUTPUT=0 disables); activation-ldb padding is on
    by default (ANTIALIAS_ACT_PAD=0 disables).
    """
    import torch._inductor.config as ind

    ind.freezing = True
    if os.environ.get("ANTIALIAS_MM_OUTPUT", "1") == "1":
        _install_offperiod_mm_output()

    _install_freezing_weight_pad()

    pad_activation = os.environ.get("ANTIALIAS_ACT_PAD", "1") == "1"
    ind.post_grad_custom_post_pass = AntiAliasStridePass(pad_activation=pad_activation)
