import sys
import pandas as pd
import time
import subprocess
import statistics
import os

def main(percentage):
    print("percentage received:", percentage, flush=True)

    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        sys.exit(1)

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    train_file = "./Train_full/training_datasets/train_full.csv"
    output_pl = "./Train_full/PL-fMDP/pl_mdp.pl"
    numeralia = "./Train_full/PL-fMDP/training_numeralia.txt"

    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    if not os.path.exists(train_file):
        print(f"Training file {train_file} does not exist.", flush=True)
        sys.exit(1)

    # -------------------------------------------------
    # Initialize numeralia
    # -------------------------------------------------
    with open(numeralia, "w") as file:
        file.write("Training time:\n")

    time_list_R = []

    # -------------------------------------------------
    # Single training run
    # -------------------------------------------------
    print("CREATE MDP (single full dataset)", flush=True)

    cmd = f"Rscript ./learning_pl-fmdps_v4_full.R {train_file} {output_pl}"
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

    print("Subprocess stdout:\n", result.stdout, flush=True)
    print("Subprocess stderr:\n", result.stderr, flush=True)

    training_time = end_time - start_time
    time_list_R.append(training_time)

    with open(numeralia, "a") as file:
        file.write(f"Training time: {training_time:.4f}\n")

    # -------------------------------------------------
    # Summary statistics (degenerate but consistent)
    # -------------------------------------------------
    average_time = statistics.mean(time_list_R)
    stdev_time = 0.0  # only one run

    with open(numeralia, "a") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

    print("Training completed successfully.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_PL-fMDPs_full.py <percentage>")
        sys.exit(1)

    percentage = sys.argv[1]
    main(percentage)

