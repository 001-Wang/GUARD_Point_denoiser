# run_all_noise_levels.py
import subprocess, sys, os, re, csv, datetime

PYTHON_EXE = sys.executable  # uses the current interpreter
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), "test_all_v5.py")

METHOD = "sngp"
# METHOD = "pointcvar"
# NUM_POINT = "2048"
GPU = "0"
KEEP = "2048"
# AMP_FLAG = "--amp"             # remove this string if you don't want AMP
TIME_POST = "--time_include_post"

# Roots to evaluate (edit as needed)
ROOTS = (
    [f"data/shapenet_c_add/add_local_s{i}" for i in range(1, 6)]
    + [f"data/shapenet_c_add/add_global_s{i}" for i in range(1, 6)]
    + [f"data/shapenet_c_add/add_ghostcluster_s{i}" for i in range(1, 6)]
)

# Output files
os.makedirs("logs", exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_LOG = os.path.join("logs", f"sngp_noamp_noise_summary_{stamp}.txt")
CSV_LOG = os.path.join("logs", f"sngp_nooamp_noise_summary_{stamp}.csv")
# TXT_LOG = os.path.join("logs", f"cvar_noamp_noise_summary_{stamp}.txt")
# CSV_LOG = os.path.join("logs", f"cvar_nooamp_noise_summary_{stamp}.csv")
# Regexes to capture summary fields from test_all_v5.py
re_latency = re.compile(
    r"Latency ms per-sample \| mean=(?P<mean>[\d.]+) \| p50=(?P<p50>[\d.]+) \| p90=(?P<p90>[\d.]+) \| p95=(?P<p95>[\d.]+) \| std=(?P<std>[\d.]+)"
)
re_noise = re.compile(r"Avg noise count in kept-(?P<keep>\d+): (?P<avg>[\d.]+)")
re_inst = re.compile(r"Instance mIoU \(kept-only, official-style\): (?P<miou>[\d.]+)")
re_class = re.compile(r"Class mIoU \(kept-only, dataset-avg\): (?P<miou>[\d.]+)")

def extract_summary(stdout_text: str):
    """
    Find the last [SUMMARY] block and parse the 4 lines we need.
    Returns dict with fields or None if parsing fails.
    """
    # Use the last occurrence of "[SUMMARY]"
    idx = stdout_text.rfind("[SUMMARY]")
    if idx == -1:
        return None
    tail = stdout_text[idx:]  # from [SUMMARY] to end
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    # Expect structure:
    # [SUMMARY]
    # Samples evaluated: N
    # Latency ms per-sample | mean=... | ...
    # Avg noise count in kept-2048: ...
    # Instance mIoU (kept-only, official-style): ...
    # Class mIoU (kept-only, dataset-avg): ...
    out = {}
    for ln in lines:
        m = re_latency.search(ln)
        if m:
            out.update({k: float(v) for k, v in m.groupdict().items()})
            continue
        m = re_noise.search(ln)
        if m:
            out["keep"] = int(m.group("keep"))
            out["avg_noise_kept"] = float(m.group("avg"))
            continue
        m = re_inst.search(ln)
        if m:
            out["inst_miou"] = float(m.group("miou"))
            continue
        m = re_class.search(ln)
        if m:
            out["class_miou"] = float(m.group("miou"))
            continue
    return out if {"mean","p50","p90","p95","std","avg_noise_kept","inst_miou","class_miou"} <= set(out) else None

def run_one(root: str):
    cmd = [
        PYTHON_EXE, "-u", TEST_SCRIPT,
        "--method", METHOD,
        "--root", root,
        "--gpu", GPU,
        "--keep", KEEP,
        TIME_POST
    ]
    # if AMP_FLAG:
    #     cmd.append(AMP_FLAG)

    print(f"\n=== Running {root} ===")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    # Append raw output to txt log (keeps everything, including per-sample lines)
    with open(TXT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {root} =====\n")
        f.write(out)
        f.write("\n")

    if proc.returncode != 0:
        print(f"[WARN] Non-zero exit for {root} (code={proc.returncode}). See {TXT_LOG} for details.")
        return None

    summary = extract_summary(out)
    if summary is None:
        print(f"[WARN] Could not parse summary for {root}. See {TXT_LOG}.")
        return None
    return summary

def main():
    print(f"Logging raw output to: {TXT_LOG}")
    print(f"Writing CSV summary to: {CSV_LOG}")

    with open(TXT_LOG, "w", encoding="utf-8") as f:
        f.write(f"==== {datetime.datetime.now()} ====\n")

    rows = []
    for root in ROOTS:
        summary = run_one(root)
        if summary:
            row = {
                "root": root,
                "keep": summary.get("keep", int(KEEP)),
                "lat_mean_ms": summary["mean"],
                "lat_p50_ms": summary["p50"],
                "lat_p90_ms": summary["p90"],
                "lat_p95_ms": summary["p95"],
                "lat_std_ms": summary["std"],
                "avg_noise_kept": summary["avg_noise_kept"],
                "inst_mIoU": summary["inst_miou"],
                "class_mIoU": summary["class_miou"],
            }
            rows.append(row)

    # Write CSV
    fieldnames = ["root","keep","lat_mean_ms","lat_p50_ms","lat_p90_ms","lat_p95_ms","lat_std_ms","avg_noise_kept","inst_mIoU","class_mIoU"]
    with open(CSV_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nDone.")
    print(f"- Raw logs: {TXT_LOG}")
    print(f"- CSV summary: {CSV_LOG}")

if __name__ == "__main__":
    main()
