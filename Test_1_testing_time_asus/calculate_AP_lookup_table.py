import sys
import pandas as pd
import time
import subprocess
import statistics
import os
import fnmatch

def main(percentage):
    print("percentage received:", percentage, flush=True)
    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        sys.exit(1)

    # Grid de hiperparámetros
    gamma_values = [0.10, 0.25, 0.50, 0.75, 0.90]
    epsilon_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

    train_path = f"./Train_{percentage}/training_datasets/"
    ap_path = f"./Train_{percentage}/models/PL-fMDP/"
    num_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print(f'Train file count: {num_files}', flush=True)

    dummyfile = f"{ap_path}dummy.pl"
    os.makedirs(os.path.dirname(dummyfile), exist_ok=True)
    cmd = f"touch {dummyfile}"
    print(cmd, flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    print("Result subprocess\n", result, flush=True)

    time_list = []  # This will store time for EACH individual AP calculation
    numeralia = f"{ap_path}training_numeralia_lookup_table.txt"
    
    with open(numeralia, "w") as file:
        file.write("AP Calculation Timings (Per Model):\n")
        file.write("=" * 50 + "\n\n")

    for i in range(1, num_files + 1):
        filename = f"{ap_path}pl_mdp_{i}.pl"
        if not os.path.exists(filename):
            print(f"Warning: File {filename} does not exist, skipping fold {i}", flush=True)
            continue
        
        fold_times = []  # Track times for this specific fold
        
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                print(f"Processing fold {i}, gamma={gamma}, epsilon={epsilon}", flush=True)
                gamma_str = str(gamma).replace('.', '_')
                epsilon_str = str(epsilon).replace('.', '_')
                apfile = f"{ap_path}APClassifier_{i}_{gamma_str}_{epsilon_str}_lookup_table.py"
                cmd = f"./mdp-problog-scikit_lookup_table solve -t 500 -m {filename} {dummyfile} -g {gamma} -e {epsilon} > {apfile}"
                print(cmd, flush=True)
                
                start_time = time.time()
                # Generates a policy in python
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
                end_time = time.time()
                print("Result subprocess\n", result, flush=True)

                training_time = end_time - start_time
                time_list.append(training_time)      # Add to overall list
                fold_times.append(training_time)     # Add to fold-specific list
                
                print(f"AP calculation time: {training_time:.4f}s", flush=True)

        # Report statistics for this fold
        with open(numeralia, "a") as file:
            file.write(f"Fold {i}:\n")
            file.write(f"  Models calculated: {len(fold_times)}\n")
            file.write(f"  Total time: {sum(fold_times):.4f}s\n")
            file.write(f"  Average per model: {statistics.mean(fold_times):.4f}s\n")
            file.write(f"  Std dev per model: {statistics.stdev(fold_times) if len(fold_times) > 1 else 0:.4f}s\n\n")

    # Calculate OVERALL statistics across ALL models
    if time_list:
        total_models = len(time_list)
        avg_time_per_model = statistics.mean(time_list)
        stdev_time_per_model = statistics.stdev(time_list) if total_models > 1 else 0
        
        with open(numeralia, "a") as file:
            file.write("OVERALL STATISTICS (ALL MODELS):\n")
            file.write("=" * 40 + "\n")
            file.write(f"Total AP models calculated: {total_models}\n")
            file.write(f"Average time per AP model: {avg_time_per_model:.4f}s\n")
            file.write(f"Std dev time per AP model: {stdev_time_per_model:.4f}s\n")
            file.write(f"Total computation time: {sum(time_list):.4f}s\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 calculate_AP_lookup_table.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
