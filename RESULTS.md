# Measured results (source machine)

Machine: AMD Ryzen AI Max+ / Radeon 8060S (`gfx1151`), 2026-09-02.

Recipe: SnapFlow 1-NFE + W4A4 prefix LM + LM fp16 + fused quant +
`torch.compile(max-autotune-no-cudagraphs)` + batched SigLIP 776 prefix.

| Metric | Value |
|--------|--------|
| E2E action chunk (compiled, closed-loop) | **~102 ms** (median ~101.6 ms) |
| Embedl W4A4 `.mxr` (reference, not this tree) | ~92 ms |
| LIBERO-10, seed 0, 2 episodes × 10 tasks | **20/20 = 100%** |
| `n_action_steps` | 10 |
| `num_inference_steps` | 1 |

Eval artifacts from the development tree (not copied here):
`onnx-infer/pi05-migraphx/eval_out/pytorch_snapflow_w4a4_libero10_ep2/`

Two episodes per task is a **smoke** for quantization/compile correctness, not a
full 500-episode LIBERO score.
