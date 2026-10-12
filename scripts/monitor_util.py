#!/usr/bin/env python3
"""Sample CPU / GPU / DRAM utilization while running a command (AMD APU, amdgpu sysfs).

  python scripts/monitor_util.py --csv eval_out/util.csv -- python scripts/bench_latency.py

Samples taken between the start/end marker lines of the child's stdout form the
"steady" window that is summarized (default markers match bench_latency.py).
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

# struct gpu_metrics_v3_0 (kgd_pp_interface.h), native alignment.
_GM_FMT = "@HBB HH16HH HH8H16H HHHH Q IHIIII 16H HHH 8H 16HHH 7I I"
_GM = {
    "gfx_activity": 22, "dram_reads": 48, "dram_writes": 49, "socket_power": 53,
    "apu_power": 55, "gfx_power": 56, "gfxclk": 78, "fclk": 82, "uclk": 84,
}
# LPDDR5X-8000 x 256-bit (Strix Halo).
DRAM_PEAK_MBPS = 256_000


def _find_gpu() -> Path:
    for p in sorted(glob.glob("/sys/class/drm/card*/device")):
        d = Path(p)
        if (d / "gpu_metrics").exists() and (d / "vendor").read_text().strip() == "0x1002":
            return d
    sys.exit("no amdgpu device with gpu_metrics found")


def _read_int(p: Path) -> int:
    return int(p.read_text().strip())


def _gpu_sample(dev: Path) -> dict:
    raw = (dev / "gpu_metrics").read_bytes()
    _, fmt_rev, content_rev = struct.unpack_from("@HBB", raw)
    s = {
        "gpu_busy_pct": _read_int(dev / "gpu_busy_percent"),
        "vram_used_mb": _read_int(dev / "mem_info_vram_used") / 2**20,
        "gtt_used_mb": _read_int(dev / "mem_info_gtt_used") / 2**20,
    }
    if (fmt_rev, content_rev) == (3, 0):
        v = struct.unpack_from(_GM_FMT, raw)
        s.update({k: v[i] for k, i in _GM.items()})
        s["dram_total_mbps"] = s["dram_reads"] + s["dram_writes"]
        s["dram_bw_pct"] = 100.0 * s["dram_total_mbps"] / DRAM_PEAK_MBPS
        for k in ("socket_power", "apu_power", "gfx_power"):
            s[k] /= 1000.0
    return s


class Sampler(threading.Thread):
    def __init__(self, pid: int, interval: float):
        super().__init__(daemon=True)
        self.root = psutil.Process(pid)
        self.interval = interval
        self.dev = _find_gpu()
        self.rows: list[dict] = []
        self.phase = "startup"
        self.stop_evt = threading.Event()
        self._procs: dict[int, psutil.Process] = {}

    def _proc_tree_cpu(self) -> tuple[float, float, float]:
        try:
            procs = [self.root, *self.root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return 0.0, 0.0, 0.0
        total = rss = 0.0
        main_cpu = 0.0
        alive = set()
        for p in procs:
            alive.add(p.pid)
            cached = self._procs.setdefault(p.pid, p)
            try:
                c = cached.cpu_percent(None)
                rss += cached.memory_info().rss
            except psutil.NoSuchProcess:
                continue
            total += c
            if p.pid == self.root.pid:
                main_cpu = c
        for pid in list(self._procs):
            if pid not in alive:
                del self._procs[pid]
        return total, main_cpu, rss / 2**20

    def run(self) -> None:
        psutil.cpu_percent(None)
        self._proc_tree_cpu()
        t0 = time.perf_counter()
        while not self.stop_evt.wait(self.interval):
            tree_cpu, main_cpu, rss = self._proc_tree_cpu()
            vm = psutil.virtual_memory()
            row = {
                "t": time.perf_counter() - t0,
                "phase": self.phase,
                "sys_cpu_pct": psutil.cpu_percent(None),
                "proc_cpu_pct": tree_cpu,
                "main_proc_cpu_pct": main_cpu,
                "proc_rss_mb": rss,
                "host_mem_used_mb": (vm.total - vm.available) / 2**20,
                "host_mem_pct": vm.percent,
            }
            row.update(_gpu_sample(self.dev))
            self.rows.append(row)


def _summarize(rows: list[dict], ncpu: int) -> None:
    if not rows:
        print("[monitor] no samples in steady window", flush=True)
        return
    keys = [k for k in rows[0] if k not in ("t", "phase")]
    print(f"\n[monitor] steady window: {len(rows)} samples, "
          f"{rows[-1]['t'] - rows[0]['t']:.1f}s, {ncpu} logical CPUs", flush=True)
    print(f"{'metric':<20}{'mean':>12}{'p50':>12}{'max':>12}")
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        print(f"{k:<20}{statistics.fmean(vals):>12.1f}{statistics.median(vals):>12.1f}{max(vals):>12.1f}")
    print("notes: proc_cpu_pct is per-core % summed over the process tree (100 = 1 core); "
          f"dram_bw_pct is vs {DRAM_PEAK_MBPS / 1000:.0f} GB/s peak; gpu_metrics are SMU "
          "time-filtered (see skip-after-start).", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--start-marker", default=r"^timed loop start")
    ap.add_argument("--end-marker", default=r"^n=\d+")
    ap.add_argument("--skip-after-start", type=float, default=2.0,
                    help="seconds dropped after start marker (SMU filter settling)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("missing command after --")

    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    s = Sampler(child.pid, a.interval)
    s.start()
    start_re, end_re = re.compile(a.start_marker), re.compile(a.end_marker)
    t_start = t_end = None
    for line in child.stdout:
        sys.stdout.write(line)
        if t_start is None and start_re.search(line):
            s.phase, t_start = "steady", time.perf_counter()
        elif t_start is not None and t_end is None and end_re.search(line):
            s.phase, t_end = "done", time.perf_counter()
    rc = child.wait()
    s.stop_evt.set()
    s.join()

    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in s.rows for k in r))
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(s.rows)
        print(f"[monitor] wrote {len(s.rows)} samples -> {a.csv}", flush=True)

    steady = [r for r in s.rows if r["phase"] == "steady"]
    if steady:
        t0 = steady[0]["t"] + a.skip_after_start
        steady = [r for r in steady if r["t"] >= t0]
    _summarize(steady, psutil.cpu_count())
    sys.exit(rc)


if __name__ == "__main__":
    main()
