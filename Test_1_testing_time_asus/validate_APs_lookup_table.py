import sys
import pandas as pd
import time
import statistics
import os
import fnmatch
import importlib.util
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_action_policy(apfile):
    """Carga dinámicamente un ActionPolicy desde un archivo .py"""
    try:
        spec = importlib.util.spec_from_file_location("ap_module", apfile)
        ap_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ap_module)

        if not hasattr(ap_module, "ActionPolicy"):
            print(f"Error: No ActionPolicy class found in {apfile}", flush=True)
            return None

        return ap_module.ActionPolicy()
    except Exception as e:
        print(f"Error loading {apfile}: {str(e)}", flush=True)
        return None

def main(percentage):
    print(f"percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"percentage as an integer: {percentage_int}", flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        sys.exit(1)

    # Grid de hiperparámetros
    gamma_values = [0.10, 0.25, 0.50, 0.75, 0.90]
    epsilon_values = [0.0001, 0.001, 0.01, 0.05, 0.1]

    validation_path = f"./Train_{percentage}/validation_datasets/"
    ap_path = f"./Train_{percentage}/models/PL-fMDP/"
    num_files = len(fnmatch.filter(os.listdir(validation_path), 'validation_fold_*.csv'))
    print(f'Validation file count: {num_files}', flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    best_params_list = []
    time_list = []
    numeralia = f"{ap_path}validation_numeralia_lookup_table.txt"
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    for i in range(1, num_files + 1):
        print(f"Validating fold {i}", flush=True)
        validation_file = f"{validation_path}validation_fold_{i}.csv"
        if not os.path.exists(validation_file):
            print(f"Validation file {validation_file} does not exist, skipping fold {i}", flush=True)
            continue

        validation_data = pd.read_csv(validation_file)
        validation_data = validation_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_val = validation_data.drop(['action'], axis=1)
        y_val = encoder.transform(validation_data['action'])

        best_f1_score = -1
        best_params = {}
        best_apfile = None

        for gamma in gamma_values:
            for epsilon in epsilon_values:
                gamma_str = str(gamma).replace('.', '_')
                epsilon_str = str(epsilon).replace('.', '_')
                apfile = f"{ap_path}APClassifier_{i}_{gamma_str}_{epsilon_str}_lookup_table.py"
                if not os.path.exists(apfile):
                    print(f"Classifier {apfile} does not exist, skipping gamma={gamma}, epsilon={epsilon}", flush=True)
                    continue

                action_policy_instance = load_action_policy(apfile)
                if action_policy_instance is None:
                    continue

                try:
                    start_time = time.time()
                    y_pred = action_policy_instance.predict(X_val)
                    end_time = time.time()

                    if y_pred is None:
                        print(f"Error: predict returned None for {apfile}", flush=True)
                        continue

                    y_pred_encoded = encoder.transform(y_pred)
                    f1 = f1_score(y_val, y_pred_encoded, average='weighted')
                    validation_time = end_time - start_time
                    time_list.append(validation_time)

                    print(f"Fold {i}, gamma={gamma}, epsilon={epsilon}, "
                          f"F1-score: {f1:.4f}, Validation time: {validation_time:.4f}", flush=True)

                    if f1 > best_f1_score:
                        best_f1_score = f1
                        best_params = {'fold': i, 'gamma': gamma, 'epsilon': epsilon, 'f1_score': f1}
                        best_apfile = apfile
                except Exception as e:
                    print(f"Error evaluating {apfile}: {str(e)}", flush=True)
                    continue

        if best_apfile:
            # Copy the best classifier to APClassifier_{i}.py
            best_output = f"{ap_path}APClassifier_{i}_lookup_table.py"
            with open(best_apfile, 'r') as src, open(best_output, 'w') as dst:
                dst.write(src.read())
            print(f"Saved best classifier for fold {i} to {best_output}", flush=True)
            best_params_list.append(best_params)

        with open(numeralia, "a") as file:
            file.write(f"Fold {i}, Best gamma: {best_params.get('gamma', 'N/A')}, "
                       f"Best epsilon: {best_params.get('epsilon', 'N/A')}, "
                       f"F1-score: {best_f1_score:.4f}\n")

    # Save best parameters to CSV
    best_params_df = pd.DataFrame(best_params_list)
    best_params_csv = f"{ap_path}best_parameters_lookup_table.csv"
    os.makedirs(os.path.dirname(best_params_csv), exist_ok=True)
    best_params_df.to_csv(best_params_csv, index=False)
    print(f"Saved best parameters to {best_params_csv}", flush=True)

    # Calculate average and standard deviation of validation times
    if time_list:
        average_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list) if len(time_list) > 1 else 0
        with open(numeralia, "a") as file:
            file.write(f"Avg. validation time: {average_time:.4f}\n")
            file.write(f"Stdev. validation time: {stdev_time:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_APs.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)

