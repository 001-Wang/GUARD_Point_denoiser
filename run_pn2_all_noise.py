# run_all_noise_levels.py  — PointNet++ batch runner for pointcvar & sngp
import subprocess, sys, os, re, csv, datetime
from pathlib import Path

# ---------- CONFIG ----------
PYTHON_EXE = sys.executable                 # use current interpreter
PROJECT_ROOT = Path(__file__).resolve().parent  # repo root where pointnet2/ lives
TEST_SCRIPT = (PROJECT_ROOT / "pointnet2" / "test_all_v6.py").as_posix()

GPU  = "0"
KEEP = "2048"
TIME_POST = True          # include selection/2nd-forward time in latency
AMP_FLAG  = False         # set True if you want --amp
# ALPHA_HYBRID = "0.7"      # used by SNGP fusion (σ² + entropy)

# Methods to run (order matters in combined CSV)
METHODS = ["pointcvar", "sngp"]
# METHODS = ["sngp"]
# Roots to evaluate (edit to your data layout)
ROOTS = (
    [f"data_prepare/shapenet_c_add/add_local_s{i}"   for i in range(1, 6)] +
    [f"data_prepare/shapenet_c_add/add_global_s{i}"  for i in range(1, 6)] +
    [f"data_prepare/shapenet_c_add/add_ghostcluster_s{i}" for i in range(1, 6)]
)
# You can also give absolute paths. Relative paths are resolved from PROJECT_ROOT.
# ---------------------------

# Regexes to parse the [SUMMARY] block printed by pointnet2/test_all_v6.py
RE_LAT   = re.compile(r"Latency ms per-sample \| mean=(?P<mean>[\d.]+) \| p50=(?P<p50>[\d.]+) \| p90=(?P<p90>[\d.]+) \| p95=(?P<p95>[\d.]+) \| std=(?P<std>[\d.]+)")
RE_NOISE = re.compile(r"Avg noise count in kept-(?P<keep>\d+): (?P<avg>[\d.]+)")
RE_INST  = re.compile(r"Instance mIoU \(kept-only, official-style\): (?P<miou>[\d.]+)")
RE_CLASS = re.compile(r"Class mIoU \(kept-only, dataset-avg\): (?P<miou>[\d.]+)")

def _as_posix(p: Path) -> str:
    return p.resolve().as_posix()

def build_cmd(method: str, root_path: str):
    """Build command to run a single job of test_all_v6.py for PointNet++."""
    cmd = [
        PYTHON_EXE, "-u", TEST_SCRIPT,
        "--method", method,
        "--root", root_path,
        "--gpu", GPU,
        "--keep", KEEP,
    ]
    if TIME_POST: cmd.append("--time_include_post")
    if AMP_FLAG:  cmd.append("--amp")
    # SNGP-only knobs (safe to pass for others but optional)
    return cmd  # <-- critical

def parse_summary(stdout_text: str):
    """Extract the last [SUMMARY] block."""
    idx = stdout_text.rfind("[SUMMARY]")
    if idx == -1:
        return None
    tail = stdout_text[idx:]
    out = {}
    for ln in (ln.strip() for ln in tail.splitlines() if ln.strip()):
        m = RE_LAT.search(ln)
        if m:
            out.update({k: float(v) for k, v in m.groupdict().items()})
            continue
        m = RE_NOISE.search(ln)
        if m:
            out["keep"] = int(m.group("keep"))
            out["avg_noise_kept"] = float(m.group("avg"))
            continue
        m = RE_INST.search(ln)
        if m:
            out["inst_miou"] = float(m.group("miou"))
            continue
        m = RE_CLASS.search(ln)
        if m:
            out["class_miou"] = float(m.group("miou"))
            continue
    req = {"mean","p50","p90","p95","std","avg_noise_kept","inst_miou","class_miou"}
    return out if req <= set(out) else None

def run_one(cmd, cwd: Path, raw_log_fp: Path):
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=_as_posix(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    with open(raw_log_fp, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write(out)
        f.write("\n")
    if proc.returncode != 0:
        print(f"[WARN] Non-zero exit code={proc.returncode}. See raw log: {raw_log_fp}")
        return None
    return parse_summary(out)

def main():
    # Resolve roots relative to repo root
    roots_abs = []
    for r in ROOTS:
        p = Path(r)
        roots_abs.append(_as_posix(p if p.is_absolute() else (PROJECT_ROOT / r)))

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-method logs & CSVs
    raw_logs = {m: logs_dir / f"pn2_{m}_{stamp}.txt" for m in METHODS}
    csv_logs = {m: logs_dir / f"pn2_{m}_{stamp}.csv" for m in METHODS}
    csv_combined = logs_dir / f"pn2_both_{stamp}.csv"

    # Clear raw logs
    for fp in raw_logs.values():
        with open(fp, "w", encoding="utf-8") as f:
            f.write(f"==== {datetime.datetime.now()} ====\n")

    fields = ["root","keep","lat_mean_ms","lat_p50_ms","lat_p90_ms","lat_p95_ms","lat_std_ms","avg_noise_kept","inst_mIoU","class_mIoU"]
    results = {m: [] for m in METHODS}

    # Run all roots for both methods
    for root in roots_abs:
        for method in METHODS:
            cmd = build_cmd(method, root)
            summary = run_one(cmd, PROJECT_ROOT, raw_logs[method])
            if summary:
                results[method].append({
                    "root": root,
                    "keep": summary.get("keep", int(KEEP)),
                    "lat_mean_ms": summary["mean"],
                    "lat_p50_ms":  summary["p50"],
                    "lat_p90_ms":  summary["p90"],
                    "lat_p95_ms":  summary["p95"],
                    "lat_std_ms":  summary["std"],
                    "avg_noise_kept": summary["avg_noise_kept"],
                    "inst_mIoU": summary["inst_miou"],
                    "class_mIoU": summary["class_miou"],
                })

    # Write per-method CSVs
    for m in METHODS:
        with open(csv_logs[m], "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results[m]:
                w.writerow(r)

    # Write combined CSV (one row per root, with both methods side-by-side when available)
    # Use the first method as the left side
    left = METHODS[0]
    right = METHODS[1] if len(METHODS) > 1 else None

    rows_comb = []
    right_by_root = {r["root"]: r for r in results.get(right, [])} if right else {}
    for lr in results[left]:
        row = dict(lr)  # base
        if right:
            rr = right_by_root.get(lr["root"])
            if rr:
                row.update({
                    f"lat_mean_ms_{right}": rr["lat_mean_ms"],
                    f"lat_p50_ms_{right}":  rr["lat_p50_ms"],
                    f"lat_p90_ms_{right}":  rr["lat_p90_ms"],
                    f"lat_p95_ms_{right}":  rr["lat_p95_ms"],
                    f"lat_std_ms_{right}":  rr["lat_std_ms"],
                    f"avg_noise_kept_{right}": rr["avg_noise_kept"],
                    f"inst_mIoU_{right}": rr["inst_mIoU"],
                    f"class_mIoU_{right}": rr["class_mIoU"],
                })
        rows_comb.append(row)

    if rows_comb:
        comb_fields = list(rows_comb[0].keys())
        with open(csv_combined, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=comb_fields)
            w.writeheader()
            for r in rows_comb:
                w.writerow(r)

    print("\nDone.")
    for m in METHODS:
        print(f"- Raw logs ({m}): {raw_logs[m]}")
        print(f"- CSV ({m})     : {csv_logs[m]}")
    if rows_comb:
        print(f"- CSV (Combined): {csv_combined}")

if __name__ == "__main__":
    main()
