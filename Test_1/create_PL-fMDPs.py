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

    train_path = f"./Train_{percentage}/training_datasets/"
    num_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print('Train file count:', num_files, flush=True)

    numeralia = f"./Train_{percentage}/models/PL-fMDP/training_numeralia.txt"
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)
    with open(numeralia, "w") as file:
        file.write("Training time:\n")

    time_list_R = []

    # Iterate over folds (1 to 20)
    for i in range(1, num_files + 1):
        print(f"CREATE MDP FOLD {i}", flush=True)
        datafile = f"{train_path}train_fold_{i}.csv"
        if not os.path.exists(datafile):
            print(f"Training file {datafile} does not exist, skipping fold {i}", flush=True)
            continue
        output = f"./Train_{percentage}/models/PL-fMDP/pl_mdp_{i}.pl"
        cmd = f"Rscript ./learning_pl-fmdps_v4.R {datafile} {output}"
        print(cmd, flush=True)
        start_time = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        end_time = time.time()
        print("Result subprocess\n", result, flush=True)

        training_time = end_time - start_time
        time_list_R.append(training_time)

        with open(numeralia, "a") as file:
            file.write(f"Fold: {i}, time: {training_time:.4f}\n")

    # Calculate average and standard deviation of training times
    if time_list_R:
        average_time = statistics.mean(time_list_R)
        stdev_time = statistics.stdev(time_list_R) if len(time_list_R) > 1 else 0
    else:
        average_time = 0
        stdev_time = 0

    with open(numeralia, "a") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_PL-fMDPs.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
