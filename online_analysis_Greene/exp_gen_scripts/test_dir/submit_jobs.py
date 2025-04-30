import os
import shutil
import yaml
import time
import subprocess
import csv
from datetime import datetime
import os

BASE_DIR = "base"
RUNS_DIR = "runs"

def submit_and_log_job(run_dir, exp_name, log_path="submitted_jobs.csv"):
    result = subprocess.run(
        ["sbatch", "mom.sub"],
        cwd=run_dir,
        capture_output=True,
        text=True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if result.returncode == 0:
        output = result.stdout.strip()
        print(f"[🚀] Submitted {exp_name}: {output}")

        if "Submitted batch job" in output:
            job_id = output.split()[-1]

            # Append to CSV log
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, exp_name, job_id, run_dir])

            return job_id
    else:
        print(f"[❌] Submission failed for {exp_name}: {result.stderr.strip()}")
        return None

def run_all_experiments_with_tracking(runs_root="runs", log_path="submitted_jobs.csv", filter_pattern=None):
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "experiment", "job_id", "path"])

    for exp_name in sorted(os.listdir(runs_root)):
        if filter_string and not re.search(filter_pattern, exp_name):
            continue  # skip experiments that don't match

        run_dir = os.path.join(runs_root, exp_name)
        momsub_path = os.path.join(run_dir, "mom.sub")

        if not os.path.isdir(run_dir) or not os.path.exists(momsub_path):
            print(f"[⚠] Skipping {exp_name} — no mom.sub found.")
            continue

        submit_and_log_job(run_dir, exp_name, log_path=log_path)


import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit SLURM jobs from experiment run directories.")
    parser.add_argument("--runs-root", type=str, default="runs", help="Directory containing experiment folders")
    parser.add_argument("--log-path", type=str, default="submitted_jobs.csv", help="Path to CSV log file")
    parser.add_argument("--filter", type=str, default=None, help="Regex pattern to match experiment names")
    parser.add_argument("--dry-run", action="store_true", help="Only print matched experiments, do not submit")

    args = parser.parse_args()

    # Prepare list of experiments
    matched_experiments = []
    for exp_name in sorted(os.listdir(args.runs_root)):
        if not os.path.isdir(os.path.join(args.runs_root, exp_name)):
            continue
        if args.filter:
            if not re.search(args.filter, exp_name):
                continue
        matched_experiments.append(exp_name)

    if args.dry_run:
        print(f"[🧪 DRY RUN] Matched {len(matched_experiments)} experiment(s):")
        for name in matched_experiments:
            print(f"  - {name}")
    else:
        if not os.path.exists(args.log_path):
            with open(args.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "experiment", "job_id", "path"])

        for name in matched_experiments:
            run_dir = os.path.join(args.runs_root, name)
            momsub_path = os.path.join(run_dir, "mom.sub")
            if not os.path.exists(momsub_path):
                print(f"[⚠] Skipping {name} — no mom.sub found.")
                continue
            submit_and_log_job(run_dir, name, log_path=args.log_path)
