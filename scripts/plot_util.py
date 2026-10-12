#!/usr/bin/env python3
"""Plot a monitor_util.py CSV as stacked time-series (phases shaded).

  python scripts/plot_util.py eval_out/util/bench_fast.csv [--phase steady] [-o out.png]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--phase", help="only plot samples of this phase (e.g. steady)")
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()

    rows = list(csv.DictReader(a.csv.open()))
    if a.phase:
        rows = [r for r in rows if r["phase"] == a.phase]
    t = [float(r["t"]) for r in rows]

    def col(k: str, scale: float = 1.0) -> list[float]:
        return [float(r[k]) * scale for r in rows]

    panels = [
        ("CPU (cores)", [("process tree", col("proc_cpu_pct", 0.01)),
                         ("system", col("sys_cpu_pct", 0.32))]),
        ("GPU busy (%)", [("gpu_busy_pct", col("gpu_busy_pct"))]),
        ("DRAM BW (GB/s)", [("read", col("dram_reads", 1e-3)), ("write", col("dram_writes", 1e-3)),
                            ("total", col("dram_total_mbps", 1e-3))]),
        ("Memory (GB)", [("host used", col("host_mem_used_mb", 1 / 1024)),
                         ("proc RSS", col("proc_rss_mb", 1 / 1024)),
                         ("GTT", col("gtt_used_mb", 1 / 1024)), ("VRAM", col("vram_used_mb", 1 / 1024))]),
        ("Power (W)", [("socket", col("socket_power")), ("gfx", col("gfx_power"))]),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.2 * len(panels)), sharex=True)
    spans: list[list] = []
    for x, r in zip(t, rows):
        if spans and spans[-1][0] == r["phase"]:
            spans[-1][2] = x
        else:
            spans.append([r["phase"], x, x])
    colors = plt.get_cmap("Pastel1").colors
    for ax, (title, series) in zip(axes, panels):
        for label, ys in series:
            ax.plot(t, ys, lw=1, label=label)
        if not a.phase:
            for i, (name, x0, x1) in enumerate(spans):
                ax.axvspan(x0, x1, color=colors[i % len(colors)], alpha=0.5, lw=0)
                if ax is axes[0]:
                    ax.text((x0 + x1) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=len(series))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{a.csv.name}" + (f" ({a.phase})" if a.phase else ""), y=1.0)
    fig.tight_layout()
    out = a.out or a.csv.with_suffix(f".{a.phase or 'all'}.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
