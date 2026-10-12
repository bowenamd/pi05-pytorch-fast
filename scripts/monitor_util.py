#!/usr/bin/env python3
"""Sample CPU / GPU / DRAM utilization while running a command (AMD APU, amdgpu sysfs).

  python scripts/monitor_util.py --csv eval_out/util.csv -- python scripts/bench_latency.py

The child's stdout lines switch the sample phase (defaults match bench_latency.py):
init -> load -> compile -> warmup -> steady -> done. A per-phase table is printed,
plus a detailed summary of the "steady" window.
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
# (phase entered, stdout regex) in order; "init" is the phase before any marker.
PHASE_MARKERS = [
    ("load", r"^loading "),
    ("compile", r"^warmup / compile"),
    ("warmup", r"^first call done"),
    ("steady", r"^timed loop start"),
    ("done", r"^n=\d+"),
]
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
        self.phase = "init"
        self.t0 = time.perf_counter()
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
        while not self.stop_evt.wait(self.interval):
            tree_cpu, main_cpu, rss = self._proc_tree_cpu()
            vm = psutil.virtual_memory()
            row = {
                "t": time.perf_counter() - self.t0,
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


def _phase_table(rows: list[dict], transitions: list[tuple[str, float]], t_end: float) -> None:
    print("\n[monitor] phases")
    print("| Phase | Start (s) | Duration (s) | CPU mean / peak (cores) | GPU busy mean (%) "
          "| DRAM mean (GB/s) | RSS end (GB) | GTT end (GB) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    bounds = transitions + [("", t_end)]
    for (name, t0), (_, t1) in zip(bounds, bounds[1:]):
        rs = [r for r in rows if r["phase"] == name]
        if not rs:
            print(f"| {name} | {t0:.1f} | {t1 - t0:.1f} | - | - | - | - | - |")
            continue
        cpu = [r["proc_cpu_pct"] / 100 for r in rs]
        print(f"| {name} | {t0:.1f} | {t1 - t0:.1f} | {statistics.fmean(cpu):.1f} / {max(cpu):.1f} "
              f"| {statistics.fmean(r['gpu_busy_pct'] for r in rs):.1f} "
              f"| {statistics.fmean(r.get('dram_total_mbps', 0) for r in rs) / 1000:.1f} "
              f"| {rs[-1]['proc_rss_mb'] / 1024:.1f} | {rs[-1]['gtt_used_mb'] / 1024:.1f} |")
    print(f"| total | 0.0 | {t_end:.1f} | | | | | |", flush=True)


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
    overrides = {"steady": a.start_marker, "done": a.end_marker}
    pending = [(name, re.compile(overrides.get(name, rx))) for name, rx in PHASE_MARKERS]
    transitions = [("init", 0.0)]
    for line in child.stdout:
        sys.stdout.write(line)
        # Markers may be skipped (other commands), but phases never go backwards.
        for i, (name, rx) in enumerate(pending):
            if rx.search(line):
                s.phase = name
                transitions.append((name, time.perf_counter() - s.t0))
                del pending[: i + 1]
                break
    rc = child.wait()
    t_end = time.perf_counter() - s.t0
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

    _phase_table(s.rows, transitions, t_end)
    steady = [r for r in s.rows if r["phase"] == "steady"]
    if steady:
        t0 = steady[0]["t"] + a.skip_after_start
        steady = [r for r in steady if r["t"] >= t0]
    _summarize(steady, psutil.cpu_count())
    sys.exit(rc)


if __name__ == "__main__":
    main()
