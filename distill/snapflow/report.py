"""Render an observed loss curve and short-run report from training evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    rows = [json.loads(line) for line in (run / "training_log.jsonl").read_text().splitlines()]
    if result["status"] != "ok" or len(rows) != result["completed_steps"]:
        raise ValueError("Only successfully completed runs can be reported")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    steps = [r["step"] for r in rows]
    for key, label in (("total", "Total"), ("flow_matching", "Flow matching"),
                       ("consistency", "Consistency")):
        values = [r[key] for r in rows]
        ax.plot(steps, values, alpha=0.3, linewidth=1)
        width = min(10, len(rows))
        means = [sum(values[max(0, i-width+1):i+1]) / min(i+1, width) for i in range(len(rows))]
        ax.plot(steps, means, label=f"{label} ({width}-step moving mean)")
    ax.set(xlabel="Training step", ylabel="Loss", title="pi0.5 SnapFlow short-run validation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run / "loss.png", dpi=160)
    plt.close(fig)
    (run / "report.md").write_text(
        "# pi0.5 SnapFlow short-run validation\n\n"
        f"Completed steps: {result['completed_steps']}. "
        f"Batch size: {manifest['arguments']['batch_size']}. "
        f"Seed: {manifest['arguments']['seed']}.\n\n"
        f"Early {result['window_steps']}-step mean loss: {result['early_mean_loss']:.6f}. "
        f"Late mean loss: {result['late_mean_loss']:.6f}. "
        f"Decrease: {result['loss_drop_pct']:.2f}%.\n\n"
        f"All recorded losses finite: {result['all_losses_finite']}. "
        f"Peak GPU allocation: {result['gpu_peak_allocated_gib']:.2f} GiB.\n\n"
        "![Observed loss](loss.png)\n\n"
        "This checks short-run training stability only. It does not establish "
        "LIBERO success rate, final convergence, or inference speedup.\n",
    )
    print(run / "report.md")


if __name__ == "__main__":
    main()
