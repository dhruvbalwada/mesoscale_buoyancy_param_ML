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


def run_all_experiments_with_tracking(runs_root="runs", log_path="submitted_jobs.csv"):
    # If the log doesn't exist yet, create it with a header
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "experiment", "job_id", "path"])

    for exp_name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, exp_name)
        momsub_path = os.path.join(run_dir, "mom.sub")

        if not os.path.isdir(run_dir) or not os.path.exists(momsub_path):
            print(f"[⚠] Skipping {exp_name} — no mom.sub found.")
            continue

        submit_and_log_job(run_dir, exp_name, log_path=log_path)

if __name__ == "__main__":
    run_all_experiments_with_tracking(
        runs_root="runs",
        log_path="submitted_jobs.csv"
    )
