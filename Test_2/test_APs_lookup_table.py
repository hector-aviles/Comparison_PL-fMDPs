import sys
import pandas as pd
import os
import fnmatch
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
import importlib.util

def load_action_policy(pyfile):
    """Carga dinámicamente un ActionPolicy desde un archivo .py"""
    try:
        spec = importlib.util.spec_from_file_location("ap_module", pyfile)
        ap_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ap_module)

        if not hasattr(ap_module, "ActionPolicy"):
            print(f"Error: No ActionPolicy class found in {pyfile}", flush=True)
            return None

        return ap_module.ActionPolicy()
    except Exception as e:
        print(f"Error loading {pyfile}: {str(e)}", flush=True)
        return None

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Percentage as an integer: {percentage_int}", flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        sys.exit(1)

    ap_path = f"./Test_{percentage}/models/PL-fMDP/"
    test_path = f"./Test_{percentage}/test_datasets/"
    numeralia = f"./Test_{percentage}/models/PL-fMDP/Results/testing_numeralia_lookup_table.txt"
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    # Find all test files
    test_files = fnmatch.filter(os.listdir(test_path), 'test_fold_*.csv')
    num_files = len(test_files)
    print(f"Test file count: {num_files}", flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    # Comprehensive metrics tracking for global and driver-level evaluation
    global_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "time": [], "test_sizes": []}
    driver_metrics = {driver: {"precision": [], "recall": [], "f1": [], "accuracy": []} for driver in range(1, 5)}

    with open(numeralia, "w") as file:
        file.write("PL-fMDP Testing Results (Lookup Table):\n")
        file.write("=" * 50 + "\n\n")

    for i in range(1, num_files + 1):
        print(f"Fold: {i}", flush=True)
        test_file = f"{test_path}test_fold_{i}.csv"
        pyfile = f"{ap_path}APClassifier_{i}_lookup_table.py"
        
        if not os.path.exists(test_file):
            print(f"Test file {test_file} does not exist, skipping fold {i}", flush=True)
            continue
        if not os.path.exists(pyfile):
            print(f"Model file {pyfile} does not exist, skipping fold {i}", flush=True)
            continue

        # Load test data with driver information
        test_data = pd.read_csv(test_file)
        test_data = test_data[["action", "driver", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_test = test_data.drop(['action'], axis=1)
        y_test = encoder.transform(test_data['action'])
        drivers = test_data['driver']
        
        test_size = len(X_test)
        global_metrics["test_sizes"].append(test_size)

        try:
            action_policy_instance = load_action_policy(pyfile)
            if action_policy_instance is None:
                continue

            # Predict on test set (without driver column for the model)
            start_time = time.time()
            X_test_no_driver = X_test.drop('driver', axis=1)
            y_pred = action_policy_instance.predict(X_test_no_driver)
            end_time = time.time()
            
            if y_pred is None:
                print(f"Error: predict returned None for {pyfile}", flush=True)
                continue
                
            test_time = end_time - start_time
            global_metrics["time"].append(test_time)
            
            y_pred_encoded = encoder.transform(y_pred)
            
            # Compute global metrics
            precision, recall, f1, accuracy = evaluate_model(y_test, y_pred_encoded)
            global_metrics["precision"].append(precision)
            global_metrics["recall"].append(recall)
            global_metrics["f1"].append(f1)
            global_metrics["accuracy"].append(accuracy)
            
            print(f"Fold {i}: Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Time: {test_time:.4f}s", flush=True)

            # Driver-level evaluation
            for driver in range(1, 5):
                driver_indices = (drivers == driver)
                if sum(driver_indices) > 0:
                    d_y_true = y_test[driver_indices]
                    d_y_pred = y_pred_encoded[driver_indices]
                    d_precision, d_recall, d_f1, d_accuracy = evaluate_model(d_y_true, d_y_pred)
                    driver_metrics[driver]["precision"].append(d_precision)
                    driver_metrics[driver]["recall"].append(d_recall)
                    driver_metrics[driver]["f1"].append(d_f1)
                    driver_metrics[driver]["accuracy"].append(d_accuracy)

            # Write detailed results for this fold
            with open(numeralia, "a") as file:
                file.write(f"Fold {i}:\n")
                file.write(f"  Test size: {test_size} rows\n")
                file.write(f"  Precision: {precision:.4f}\n")
                file.write(f"  Recall: {recall:.4f}\n")
                file.write(f"  F1-score: {f1:.4f}\n")
                file.write(f"  Accuracy: {accuracy:.4f}\n")
                file.write(f"  Test time: {test_time:.4f}s\n")
                file.write(f"  Time per row: {test_time/test_size:.6f}s\n")
                file.write(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_encoded)}\n")
                file.write(f"  Classification Report:\n{classification_report(y_test, y_pred_encoded)}\n\n")
                
        except Exception as e:
            print(f"Error testing fold {i}: {str(e)}", flush=True)
            continue

    # Calculate and write overall statistics
    if global_metrics["time"]:
        avg_size = statistics.mean(global_metrics["test_sizes"]) if global_metrics["test_sizes"] else 0
        
        with open(numeralia, "a") as file:
            file.write("OVERALL TESTING STATISTICS:\n")
            file.write("=" * 50 + "\n")
            file.write(f"Total tests: {len(global_metrics['time'])}\n")
            file.write(f"Average test size: {avg_size:.1f} rows\n")
            file.write(f"Average test time: {statistics.mean(global_metrics['time']):.4f}s\n")
            file.write(f"Std dev test time: {statistics.stdev(global_metrics['time']) if len(global_metrics['time']) > 1 else 0:.4f}s\n")
            file.write(f"Average time per row: {statistics.mean(global_metrics['time'])/avg_size:.6f}s\n")
            
            # Global metrics
            for metric in ["precision", "recall", "f1", "accuracy"]:
                avg = statistics.mean(global_metrics[metric])
                stdev = statistics.stdev(global_metrics[metric]) if len(global_metrics[metric]) > 1 else 0
                file.write(f"Average {metric.capitalize()}: {avg:.4f} ± {stdev:.4f}\n")
            
            # Driver-level metrics
            file.write("\nDRIVER-LEVEL METRICS:\n")
            file.write("-" * 30 + "\n")
            for driver in range(1, 5):
                file.write(f"Driver {driver}:\n")
                for metric in ["precision", "recall", "f1", "accuracy"]:
                    if driver_metrics[driver][metric]:
                        avg = statistics.mean(driver_metrics[driver][metric])
                        stdev = statistics.stdev(driver_metrics[driver][metric]) if len(driver_metrics[driver][metric]) > 1 else 0
                        file.write(f"  {metric.capitalize()}: {avg:.4f} ± {stdev:.4f}\n")
                    else:
                        file.write(f"  {metric.capitalize()}: No data\n")
                file.write("\n")

        # Save driver F1 scores for plotting (matching the second code's format)
        os.makedirs("./Images", exist_ok=True)
        for driver in range(1, 5):
            if driver_metrics[driver]["f1"]:
                driver_f1_avg = statistics.mean(driver_metrics[driver]["f1"])
                driver_f1_stdev = statistics.stdev(driver_metrics[driver]["f1"]) if len(driver_metrics[driver]["f1"]) > 1 else 0
                driver_file = f"./Images/driver_{driver}_lookup_table.txt"
                with open(driver_file, "w") as f:
                    f.write(f"PL-fMDP (Lookup Table):\n")
                    f.write(f"Mean F1 Score: {driver_f1_avg}\nF1 Score Std Dev:  {driver_f1_stdev}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_APs_lookup_table.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
