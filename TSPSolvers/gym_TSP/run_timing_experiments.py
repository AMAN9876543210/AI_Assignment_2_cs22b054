#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Written by Aman Anand on 2025-04-05
# This script executes TSP optimization scripts and logs their performance metrics.
# -----------------------------------------------------------------------------

import os
import subprocess
import time
import csv
import re

# List of TSP solver script filenames to execute.
script_files = ["hill_climbing.py", "simulated_annealing.py"]

# Output filenames for the CSV summary and detailed log.
results_csv = "results_summary.csv"
log_file_path = "execution_details.log"
num_iterations = 5

# Prepare the environment with proper encoding.
env_vars = os.environ.copy()
env_vars["PYTHONIOENCODING"] = "utf-8"

all_results = []

# Define metric keys to extract from each script's output.
metric_names = ["Best Distance", "Time", "Convergence Point", "Reward"]

def extract_metrics_from_output(output_text):
    """Extracts performance metrics from the provided script output."""
    extracted = {key: "N/A" for key in metric_names}
    try:
        for line in output_text.splitlines():
            for metric in metric_names:
                if metric in line:
                    match = re.search(rf"{metric}[:\s]+(-?\d+\.?\d*)", line)
                    if match:
                        extracted[metric] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
    except Exception as err:
        print(f"⚠️ Issue extracting metrics: {err}")
    return extracted

with open(log_file_path, "w", encoding="utf-8") as log_handle:

    def execute_script(script_path):
        start = time.time()
        try:
            proc_result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=1200,
                encoding="utf-8",
                errors="replace",
                env=env_vars
            )
            elapsed = time.time() - start
            stdout_text = proc_result.stdout.strip()
            stderr_text = proc_result.stderr.strip()

            log_handle.write(f"\n=== Output from {script_path} ===\n{stdout_text}\n")
            if stderr_text:
                log_handle.write(f"\n⚠️ Errors:\n{stderr_text}\n")
            log_handle.write("===============================\n")

            metrics_data = extract_metrics_from_output(stdout_text)
            metrics_data["Execution Time"] = elapsed
            return metrics_data

        except subprocess.TimeoutExpired:
            log_handle.write(f"⚠️ {script_path} exceeded the time limit!\n")
            return {key: "Timeout" for key in metric_names + ["Execution Time"]}
        except Exception as exc:
            log_handle.write(f"❌ Error executing {script_path}: {exc}\n")
            return {key: "Error" for key in metric_names + ["Execution Time"]}

    for iter_count in range(num_iterations):
        print(f"Running test {iter_count + 1}/{num_iterations}...")
        log_handle.write(f"\n🔄 Starting test {iter_count + 1}/{num_iterations}...\n")

        row_data = [iter_count + 1]
        for script in script_files:
            metrics = execute_script(script)
            row_data.extend([metrics[m] for m in metric_names] + [metrics["Execution Time"]])
        all_results.append(row_data)

# Create header row for the CSV file.
csv_header = ["Iteration"]
for script in script_files:
    csv_header += [f"{script} - {metric}" for metric in metric_names + ["Execution Time"]]

with open(results_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)
    writer.writerows(all_results)

print(f"✅ Summary results saved in {results_csv}")
print(f"📜 Detailed log available in {log_file_path}")
print("🔍 Please refer to the log file for full output details.")

