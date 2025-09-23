import sys
import pandas as pd
import os
import fnmatch
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
from action_policy import ActionPolicy
import ast
import types

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def extract_predict_method(file_path):
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ActionPolicy':
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == 'predict':
                        predict_source = ast.get_source_segment(source, method)
                        print(f"Extracted predict method from {file_path}:\n{predict_source}", flush=True)
                        return predict_source
        print(f"No predict method found in {file_path}", flush=True)
        return None
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}", flush=True)
        return None

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Percentage as an integer: {percentage_int}", flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        return
    
    test_path = f"./Test_{percentage}/test_datasets/"
    ap_path = f"./Test_{percentage}/models/PL-fMDP/"
    
    # MODIFIED BLOCK START: Added output file path for testing_numeralia.txt
    output = f"./Test_{percentage}/models/PL-fMDP/Results/testing_numeralia.txt"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # MODIFIED BLOCK END

    num_files = len(fnmatch.filter(os.listdir(test_path), 'test_fold_*.csv'))
    print('Test file count:', num_files)
    
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])
    
    # MODIFIED BLOCK START: Added comprehensive metrics tracking to match NB format
    global_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "time": []}
    driver_metrics = {driver: {"precision": [], "recall": [], "f1": [], "accuracy": []} for driver in range(1, 5)}
    
    with open(output, "w") as file:
        file.write("Testing data\n")
    # MODIFIED BLOCK END

    for i in range(1, num_files + 1):  # MODIFIED: Changed to num_files + 1 to match NB
        print(f"Fold: {i}", flush=True)
        filename = f"./Test_{percentage}/test_datasets/test_fold_{i}.csv"
        if not os.path.exists(filename):
            print(f"Test file {filename} does not exist, skipping fold {i}", flush=True)
            continue
        
        data = pd.read_csv(filename)
        data = data[["action", "driver", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        y_true = encoder.transform(data['action'])
        X_test = data.drop(columns=['action'])
        drivers = data['driver']

        pyfile = f"{ap_path}APClassifier_{i}.py"
        if not os.path.exists(pyfile):
            print(f"Py file {pyfile} does not exist, skipping fold {i}", flush=True)
            continue

        try:
            # Extract and bind predict method
            predict_source = extract_predict_method(pyfile)
            if predict_source is None:
                print(f"No valid predict method in {pyfile}, skipping", flush=True)
                continue
            action_policy_instance = ActionPolicy()
            method_code = compile(predict_source, pyfile, 'exec')
            method_globals = {'pd': pd}
            exec(method_code, method_globals)
            if 'predict' not in method_globals:
                print(f"Error: predict method not defined in {pyfile}", flush=True)
                continue
            action_policy_instance.predict = types.MethodType(method_globals['predict'], action_policy_instance)

            # Predict on full test set (without driver column)
            start_time = time.time()
            X_test_no_driver = X_test.drop('driver', axis=1)
            y_pred = action_policy_instance.predict(X_test_no_driver)
            end_time = time.time()
            
            if y_pred is None:
                print(f"Error: predict returned None for {pyfile}", flush=True)
                continue
            
            y_pred_encoded = encoder.transform(y_pred)
            
            # MODIFIED BLOCK START: Added timing and comprehensive evaluation to match NB
            global_metrics["time"].append(end_time - start_time)
            
            precision, recall, f1, accuracy = evaluate_model(y_true, y_pred_encoded)
            global_metrics["precision"].append(precision)
            global_metrics["recall"].append(recall)
            global_metrics["f1"].append(f1)
            global_metrics["accuracy"].append(accuracy)
            
            with open(output, "a") as file:
                file.write(f"Fold: {i}\n{confusion_matrix(y_true, y_pred_encoded)}\n")
                file.write(classification_report(y_true, y_pred_encoded) + "\n")
            # MODIFIED BLOCK END
            
            # MODIFIED BLOCK START: Changed driver metrics processing to match NB format
            for driver in range(1, 5):
                driver_indices = (drivers == driver)
                if sum(driver_indices) > 0:
                    d_y_true = y_true[driver_indices]
                    d_y_pred = y_pred_encoded[driver_indices]
                    d_precision, d_recall, d_f1, d_accuracy = evaluate_model(d_y_true, d_y_pred)
                    driver_metrics[driver]["precision"].append(d_precision)
                    driver_metrics[driver]["recall"].append(d_recall)
                    driver_metrics[driver]["f1"].append(d_f1)
                    driver_metrics[driver]["accuracy"].append(d_accuracy)
            # MODIFIED BLOCK END
                
        except Exception as e:
            print(f"Error evaluating {pyfile}: {str(e)}", flush=True)
            continue

    # MODIFIED BLOCK START: Write comprehensive results to match NB format exactly
    with open(output, "a") as file:
        for metric in ["precision", "recall", "f1", "accuracy", "time"]:
            avg = statistics.mean(global_metrics[metric])
            stdev = statistics.stdev(global_metrics[metric]) if len(global_metrics[metric]) > 1 else 0
            file.write(f"Average {metric.capitalize()}: {avg}\nStd Dev {metric.capitalize()}: {stdev}\n")
        
        for driver in range(1, 5):
            file.write(f"\nDriver {driver} Metrics:\n")
            for metric in ["precision", "recall", "f1", "accuracy"]:
                avg = statistics.mean(driver_metrics[driver][metric]) if driver_metrics[driver][metric] else 0
                stdev = statistics.stdev(driver_metrics[driver][metric]) if len(driver_metrics[driver][metric]) > 1 else 0
                file.write(f"{metric.capitalize()}: {avg}, Std Dev: {stdev}\n")
    # MODIFIED BLOCK END
    
    # MODIFIED BLOCK START: Also save F1 scores to driver files for plotting to match NB format
    os.makedirs("./Images", exist_ok=True)
    
    for driver in range(1, 5):
        if driver_metrics[driver]["f1"]:
            driver_f1_avg = statistics.mean(driver_metrics[driver]["f1"])
            driver_f1_stdev = statistics.stdev(driver_metrics[driver]["f1"]) if len(driver_metrics[driver]["f1"]) > 1 else 0
            driver_file = f"./Images/driver_{driver}.txt"
            with open(driver_file, "a") as f:
                f.write(f"PL-fMDP:\n")
                f.write(f"Mean F1 Score: {driver_f1_avg}\nF1 Score Std Dev:  {driver_f1_stdev}\n")
    # MODIFIED BLOCK END

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_PL-fMDP.py <percentage>")
        sys.exit(1)
    main(sys.argv[1])
