import sys
import pandas as pd
import time
import subprocess
import statistics
import os

def main():

    # -------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------
    gamma_values = [0.10]
    epsilon_values = [0.0001]

    # -------------------------------------------------
    # Paths (single dataset / single model)
    # -------------------------------------------------
    train_path = "./Train_full/training_datasets/"
    ap_path = "./Train_full/PL-fMDP/"

    train_file = os.path.join(train_path, "train_full.csv")
    pl_file = os.path.join(ap_path, "pl_mdp.pl")

    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}", flush=True)
        sys.exit(1)

    if not os.path.exists(pl_file):
        print(f"ProbLog model not found: {pl_file}", flush=True)
        sys.exit(1)

    print("Using training file:", train_file, flush=True)
    print("Using ProbLog model:", pl_file, flush=True)

    # -------------------------------------------------
    # Dummy file (required by mdp-problog)
    # -------------------------------------------------
    dummyfile = f"{ap_path}dummy.pl"
    os.makedirs(os.path.dirname(dummyfile), exist_ok=True)
    subprocess.run(f"touch {dummyfile}", shell=True)

    # -------------------------------------------------
    # Timing and logging
    # -------------------------------------------------
    time_list = []
    numeralia = f"{ap_path}training_numeralia_lookup_table.txt"

    with open(numeralia, "w") as file:
        file.write("AP Calculation Timings (Single Model):\n")
        file.write("=" * 50 + "\n\n")

    # -------------------------------------------------
    # Run AP calculation once (per hyperparameter pair)
    # -------------------------------------------------
    for gamma in gamma_values:
        for epsilon in epsilon_values:
            print(f"Processing FULL dataset, gamma={gamma}, epsilon={epsilon}", flush=True)

            gamma_str = str(gamma).replace('.', '_')
            epsilon_str = str(epsilon).replace('.', '_')

            apfile = f"{ap_path}APClassifier_full_{gamma_str}_{epsilon_str}_lookup_table.py"

            cmd = (
                f"./mdp-problog-scikit_lookup_table solve "
                f"-t 500 "
                f"-m {pl_file} "
                f"{dummyfile} "
                f"-g {gamma} "
                f"-e {epsilon} "
                f"> {apfile}"
            )

            print(cmd, flush=True)

            start_time = time.time()
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
            )
            end_time = time.time()

            print("Result subprocess\n", result, flush=True)

            training_time = end_time - start_time
            time_list.append(training_time)

            print(f"AP calculation time: {training_time:.4f}s", flush=True)

    # -------------------------------------------------
    # Report statistics
    # -------------------------------------------------
    if time_list:
        total_models = len(time_list)
        avg_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list) if total_models > 1 else 0

        with open(numeralia, "a") as file:
            file.write("FULL DATASET RESULTS:\n")
            file.write("=" * 40 + "\n")
            file.write(f"Models calculated: {total_models}\n")
            file.write(f"Total computation time: {sum(time_list):.4f}s\n")
            file.write(f"Average time per model: {avg_time:.4f}s\n")
            file.write(f"Std dev time per model: {stdev_time:.4f}s\n")

    print("AP calculation completed successfully.", flush=True)


if __name__ == "__main__":
    main()

