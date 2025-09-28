import sys
import pandas as pd
import os
import fnmatch
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Percentage as an integer: {percentage_int}", flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        sys.exit(1)

    ap_path = f"./Train_{percentage}/models/PL-fMDP/"
    test_path = f"./Train_{percentage}/test_datasets/"
    numeralia = f"./Train_{percentage}/models/PL-fMDP/Results/testing_numeralia_lookup_table.txt"
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    # Find all test files
    test_files = fnmatch.filter(os.listdir(test_path), 'test_fold_*.csv')
    num_files = len(test_files)
    print(f"Test file count: {num_files}", flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    precisions, recalls, f1_scores, accuracies, test_times, test_sizes = [], [], [], [], [], []

    with open(numeralia, "w") as file:
        file.write("PL-fMDP Testing Results:\n")
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

        # Load test data
        test_data = pd.read_csv(test_file)
        test_data = test_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_test = test_data.drop(['action'], axis=1)
        y_test = encoder.transform(test_data['action'])
        
        test_size = len(X_test)
        test_sizes.append(test_size)

        try:
            action_policy_instance = load_action_policy(pyfile)
            if action_policy_instance is None:
                continue

            # Predict on test set
            start_time = time.time()
            y_pred = action_policy_instance.predict(X_test)
            end_time = time.time()
            
            if y_pred is None:
                print(f"Error: predict returned None for {pyfile}", flush=True)
                continue
                
            test_time = end_time - start_time
            test_times.append(test_time)
            
            y_pred_encoded = encoder.transform(y_pred)
            
            # Compute metrics
            precision = precision_score(y_test, y_pred_encoded, average='weighted')
            recall = recall_score(y_test, y_pred_encoded, average='weighted')
            f1 = f1_score(y_test, y_pred_encoded, average='weighted')
            accuracy = accuracy_score(y_test, y_pred_encoded)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            accuracies.append(accuracy)
            
            print(f"Fold {i}: Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Time: {test_time:.4f}s", flush=True)

            with open(numeralia, "a") as file:
                file.write(f"Fold {i}:\n")
                file.write(f"  Test size: {test_size} rows\n")
                file.write(f"  Precision: {precision:.4f}\n")
                file.write(f"  Recall: {recall:.4f}\n")
                file.write(f"  F1-score: {f1:.4f}\n")
                file.write(f"  Accuracy: {accuracy:.4f}\n")
                file.write(f"  Test time: {test_time:.4f}s\n")
                file.write(f"  Time per row: {test_time/test_size:.6f}s\n\n")
                
        except Exception as e:
            print(f"Error testing fold {i}: {str(e)}", flush=True)
            continue

    # Calculate average metrics
    if test_times:
        metrics = {
            'precision': (statistics.mean(precisions), statistics.stdev(precisions) if len(precisions) > 1 else 0),
            'recall': (statistics.mean(recalls), statistics.stdev(recalls) if len(recalls) > 1 else 0),
            'f1': (statistics.mean(f1_scores), statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0),
            'accuracy': (statistics.mean(accuracies), statistics.stdev(accuracies) if len(accuracies) > 1 else 0),
            'time': (statistics.mean(test_times), statistics.stdev(test_times) if len(test_times) > 1 else 0)
        }
        
        avg_size = statistics.mean(test_sizes) if test_sizes else 0
        
        with open(numeralia, "a") as file:
            file.write("OVERALL TESTING STATISTICS:\n")
            file.write("=" * 40 + "\n")
            file.write(f"Total tests: {len(test_times)}\n")
            file.write(f"Average test size: {avg_size:.1f} rows\n")
            file.write(f"Average test time: {metrics['time'][0]:.4f}s\n")
            file.write(f"Std dev test time: {metrics['time'][1]:.4f}s\n")
            file.write(f"Average time per row: {metrics['time'][0]/avg_size:.6f}s\n")
            file.write(f"Average F1-score: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}\n")
            file.write(f"Average Precision: {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}\n")
            file.write(f"Average Recall: {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}\n")
            file.write(f"Average Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_APs_lookup_table.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)

