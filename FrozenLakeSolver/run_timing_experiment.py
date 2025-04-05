#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Written by aman anand on 2025-04-05
# This script runs Frozen Lake solver experiments (BnB and IDA*) multiple times,
# logs the outputs, and saves the execution times into a CSV file.
# -----------------------------------------------------------------------------

import subprocess
import time
import csv

# Define file paths
bnb_file = "bnb_frozen_lake.py"
ida_file = "ida_frozen_lake.py"
csv_file = "execution_times.csv"
log_file_name = "execution_log.txt"

num_experiments = 5
experiment_results = []

# Open the log file for writing details of each run
with open(log_file_name, "w", encoding="utf-8") as log_handle:

    def execute_script(script_path):
        """Executes a given script, logs its output, and extracts its execution time."""
        start = time.time()
        proc = subprocess.run(["python3", script_path], capture_output=True, text=True)
        run_duration = time.time() - start

        output_text = proc.stdout.strip()
        log_handle.write(f"\n=== Output from {script_path} ===\n{output_text}\n=============================\n")

        try:
            # Extract the execution time from the script's output
            exec_time = float(output_text.split("Execution Time: ")[1].split(" seconds")[0])
            return exec_time
        except (IndexError, ValueError) as err:
            log_handle.write(f"⚠️ Parsing error in {script_path}: {err}\n")
            return float("inf")

    # Run experiments multiple times
    for test_num in range(num_experiments):
        log_handle.write(f"\nRunning experiment {test_num+1}/{num_experiments}...\n")
        print(f"Running experiment {test_num+1}/{num_experiments}...")

        bnb_exec_time = execute_script(bnb_file)
        ida_exec_time = execute_script(ida_file)

        experiment_results.append([test_num + 1, bnb_exec_time, ida_exec_time])

# Save the results to a CSV file
with open(csv_file, mode="w", newline="", encoding="utf-8") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["Iteration", "BnB Execution Time (s)", "IDA* Execution Time (s)"])
    writer.writerows(experiment_results)

print(f"✅ Execution times saved in {csv_file}")
print(f"📜 Full log saved in {log_file_name}")

