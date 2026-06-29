# run_dgcnn_benchmark.py
import subprocess, sys, os, re, csv, datetime
from pathlib import Path

# --- CONFIG (edit these if your paths differ) ---
PROJECT_ROOT = Path(r"D:\zuoxu\pointsngp")  # where `dgcnn/` and `pointnet2/` live (module root)
NOISE_BASE   = Path(r"D:\zuoxu\Pointnet2\data\shapenet_c_add")  # folder containing add_* subfolders

# SNGP PN-front (absolute paths)
PNFRONT_MODULE   = "pointnet2.models.sngp_s2_6layers"
PNFRONT_CKPT_ABS = (PROJECT_ROOT / "pointnet2/log/sngp/checkpoints/best_model.ckpt").resolve()
P_CLEAN_ABS      = (PROJECT_ROOT / "pointnet2/log/sngp/checkpoints/P_clean.pt").resolve()

# Common run settings
GPU      = "0"
KEEP     = "2048"
TIMEPOST = True
AMP_FLAG = False  # set True if you want --amp

# Optional: restrict which subfolders under NOISE_BASE to run; None = all dirs starting with "add_"
# Example: ONLY_FOLDERS = ["add_ghostcluster_s5", "add_local_s5"]
ONLY_FOLDERS = None
# ------------------------------------------------


def _as_posix(p: Path) -> str:
    return p.resolve().as_posix()

def discover_roots(base: Path):
    """Return absolute POSIX paths to all noise roots under base."""
    items = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("add_"):
            continue
        if ONLY_FOLDERS and name not in ONLY_FOLDERS:
            continue
        items.append(child.resolve())
    if not items:
        print(f"[WARN] No noise folders found under: {base}")
    return items

def build_cmd_pointcvar(pyexe: str, root_abs: Path):
    cmd = [
        pyexe, "-u", "-m", "dgcnn.test_all_dgcnn_unified",
        "--method", "pointcvar",
        "--root", _as_posix(root_abs),
        "--gpu", GPU,
        "--keep", KEEP,
    ]
    if TIMEPOST: cmd.append("--time_include_post")
    if AMP_FLAG: cmd.append("--amp")
    return cmd

def build_cmd_sngp(pyexe: str, root_abs: Path):
    cmd = [
        pyexe, "-u", "-m", "dgcnn.test_all_dgcnn_unified",
        "--method", "sngp",
        "--pnfront_module", PNFRONT_MODULE,
        "--pnfront_ckpt", _as_posix(PNFRONT_CKPT_ABS),
        "--precision_path", _as_posix(P_CLEAN_ABS),
        "--root", _as_posix(root_abs),
        "--gpu", GPU,
        "--keep", KEEP,
    ]
    if TIMEPOST: cmd.append("--time_include_post")
    if AMP_FLAG: cmd.append("--amp")
    return cmd

# Regex to extract the [SUMMARY] block fields your scripts print
RE_LAT   = re.compile(r"Latency ms per-sample \| mean=(?P<mean>[\d.]+) \| p50=(?P<p50>[\d.]+) \| p90=(?P<p90>[\d.]+) \| p95=(?P<p95>[\d.]+) \| std=(?P<std>[\d.]+)")
RE_NOISE = re.compile(r"Avg noise count in kept-(?P<keep>\d+): (?P<avg>[\d.]+)")
RE_INST  = re.compile(r"Instance mIoU \(kept-only, official-style\): (?P<miou>[\d.]+)")
RE_CLASS = re.compile(r"Class mIoU \(kept-only, dataset-avg\): (?P<miou>[\d.]+)")

def parse_summary(text: str):
    idx = text.rfind("[SUMMARY]")
    if idx == -1:
        return None
    tail = text[idx:]
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
    required = {"mean","p50","p90","p95","std","avg_noise_kept","inst_miou","class_miou"}
    return out if required <= set(out) else None

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
    pyexe = sys.executable
    roots = discover_roots(NOISE_BASE)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_log_pointcvar = logs_dir / f"dgcnn_pointcvar_{stamp}.txt"
    raw_log_sngp      = logs_dir / f"dgcnn_sngp_{stamp}.txt"
    csv_pointcvar     = logs_dir / f"dgcnn_pointcvar_{stamp}.csv"
    csv_sngp          = logs_dir / f"dgcnn_sngp_{stamp}.csv"
    csv_combined      = logs_dir / f"dgcnn_both_{stamp}.csv"

    # Prepare CSV writers
    fields = ["root","keep","lat_mean_ms","lat_p50_ms","lat_p90_ms","lat_p95_ms","lat_std_ms","avg_noise_kept","inst_mIoU","class_mIoU"]
    def write_csv(path: Path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow(r)

    # Run both methods
    rows_pointcvar, rows_sngp = [], []
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Noise base  : {NOISE_BASE}")
    print(f"[INFO] Found {len(roots)} noise folders to evaluate.\n")

    for root_abs in roots:
        # PointCVaR
        pc_cmd = build_cmd_pointcvar(pyexe, root_abs)
        pc_sum = run_one(pc_cmd, PROJECT_ROOT, raw_log_pointcvar)
        if pc_sum:
            rows_pointcvar.append({
                "root": _as_posix(root_abs),
                "keep": pc_sum.get("keep", int(KEEP)),
                "lat_mean_ms": pc_sum["mean"],
                "lat_p50_ms":  pc_sum["p50"],
                "lat_p90_ms":  pc_sum["p90"],
                "lat_p95_ms":  pc_sum["p95"],
                "lat_std_ms":  pc_sum["std"],
                "avg_noise_kept": pc_sum["avg_noise_kept"],
                "inst_mIoU": pc_sum["inst_miou"],
                "class_mIoU": pc_sum["class_miou"],
            })

        # SNGP
        s_cmd = build_cmd_sngp(pyexe, root_abs)
        s_sum = run_one(s_cmd, PROJECT_ROOT, raw_log_sngp)
        if s_sum:
            rows_sngp.append({
                "root": _as_posix(root_abs),
                "keep": s_sum.get("keep", int(KEEP)),
                "lat_mean_ms": s_sum["mean"],
                "lat_p50_ms":  s_sum["p50"],
                "lat_p90_ms":  s_sum["p90"],
                "lat_p95_ms":  s_sum["p95"],
                "lat_std_ms":  s_sum["std"],
                "avg_noise_kept": s_sum["avg_noise_kept"],
                "inst_mIoU": s_sum["inst_miou"],
                "class_mIoU": s_sum["class_miou"],
            })

    # Write individual CSVs
    write_csv(csv_pointcvar, rows_pointcvar)
    write_csv(csv_sngp, rows_sngp)

    # Combined CSV (matched by root where both exist)
    combined_rows = []
    sngp_by_root = {r["root"]: r for r in rows_sngp}
    for pr in rows_pointcvar:
        sr = sngp_by_root.get(pr["root"])
        row = {
            "root": pr["root"],
            "keep": pr["keep"],
            "lat_mean_ms": pr["lat_mean_ms"],
            "lat_p50_ms": pr["lat_p50_ms"],
            "lat_p90_ms": pr["lat_p90_ms"],
            "lat_p95_ms": pr["lat_p95_ms"],
            "lat_std_ms": pr["lat_std_ms"],
            "avg_noise_kept": pr["avg_noise_kept"],
            "inst_mIoU": pr["inst_mIoU"],
            "class_mIoU": pr["class_mIoU"],
        }
        if sr:
            # append SNGP fields with suffix
            row.update({
                "lat_mean_ms_sngp": sr["lat_mean_ms"],
                "lat_p50_ms_sngp":  sr["lat_p50_ms"],
                "lat_p90_ms_sngp":  sr["lat_p90_ms"],
                "lat_p95_ms_sngp":  sr["lat_p95_ms"],
                "lat_std_ms_sngp":  sr["lat_std_ms"],
                "avg_noise_kept_sngp": sr["avg_noise_kept"],
                "inst_mIoU_sngp": sr["inst_mIoU"],
                "class_mIoU_sngp": sr["class_mIoU"],
            })
        combined_rows.append(row)

    # Save combined CSV
    comb_fields = list(combined_rows[0].keys()) if combined_rows else []
    if combined_rows:
        with open(csv_combined, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=comb_fields)
            w.writeheader()
            for r in combined_rows: w.writerow(r)

    print("\nDone.")
    print(f"- Raw logs (PointCVaR): {raw_log_pointcvar}")
    print(f"- Raw logs (SNGP)     : {raw_log_sngp}")
    print(f"- CSV (PointCVaR)     : {csv_pointcvar}")
    print(f"- CSV (SNGP)          : {csv_sngp}")
    if combined_rows:
        print(f"- CSV (Combined)      : {csv_combined}")

if __name__ == "__main__":
    main()
