import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import pickle
import time
import statistics
import numpy as np
import os
import fnmatch
from sklearn.preprocessing import LabelEncoder

def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def main(percentage):
    print("percentage received:", percentage, flush=True)
    
    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        return
    
    test_path = f"./Test_{percentage}/test_datasets/"
    num_files = len(fnmatch.filter(os.listdir(test_path), 'test_fold_*.csv'))
    print('Test file count:', num_files)
    
    # MODIFIED BLOCK START: Added output file path for testing_numeralia.txt
    output = f"./Test_{percentage}/models/RF/Results/testing_numeralia.txt"
    # MODIFIED BLOCK END
    
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])
    
    # MODIFIED BLOCK START: Added comprehensive metrics tracking
    global_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "time": []}
    driver_metrics = {driver: {"precision": [], "recall": [], "f1": [], "accuracy": []} for driver in range(1, 5)}
    
    with open(output, "w") as file:
        file.write("Testing data\n")
    # MODIFIED BLOCK END
    
    for i in range(1, num_files + 1):
        print("Fold:", i, flush=True)
        filename = f"./Test_{percentage}/test_datasets/test_fold_{i}.csv"
        data = pd.read_csv(filename)
        y_true = encoder.transform(data['action'])
        X_test = data.drop(columns=['action', 'driver'])
        drivers = data['driver']
        
        model_filename = f"./Test_{percentage}/models/RF/RF_{i}.rf"
        model = pickle.load(open(model_filename, 'rb'))
        
        # MODIFIED BLOCK START: Added timing and comprehensive evaluation
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        
        global_metrics["time"].append(end_time - start_time)
        
        precision, recall, f1, accuracy = evaluate_model(y_true, y_pred)
        global_metrics["precision"].append(precision)
        global_metrics["recall"].append(recall)
        global_metrics["f1"].append(f1)
        global_metrics["accuracy"].append(accuracy)
        
        with open(output, "a") as file:
            file.write(f"Fold: {i}\n{confusion_matrix(y_true, y_pred)}\n")
            file.write(classification_report(y_true, y_pred) + "\n")
        # MODIFIED BLOCK END
        
        for driver in range(1, 5):
            driver_indices = (drivers == driver)
            if sum(driver_indices) > 0:
                d_y_true = y_true[driver_indices]
                d_y_pred = y_pred[driver_indices]
                d_precision, d_recall, d_f1, d_accuracy = evaluate_model(d_y_true, d_y_pred)
                # MODIFIED BLOCK START: Store all metrics for each driver
                driver_metrics[driver]["precision"].append(d_precision)
                driver_metrics[driver]["recall"].append(d_recall)
                driver_metrics[driver]["f1"].append(d_f1)
                driver_metrics[driver]["accuracy"].append(d_accuracy)
                # MODIFIED BLOCK END
    
    # MODIFIED BLOCK START: Write comprehensive results to testing_numeralia.txt
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
    
    # MODIFIED BLOCK START: Also save F1 scores to driver files for plotting
    os.makedirs("./Images", exist_ok=True)
    
    for driver in range(1, 5):
        if driver_metrics[driver]["f1"]:
            driver_f1_avg = statistics.mean(driver_metrics[driver]["f1"])
            driver_f1_stdev = statistics.stdev(driver_metrics[driver]["f1"]) if len(driver_metrics[driver]["f1"]) > 1 else 0
            driver_file = f"./Images/driver_{driver}.txt"
            with open(driver_file, "a") as f:
                f.write(f"RF:\n")
                f.write(f"Mean F1 Score: {driver_f1_avg}\nF1 Score Std Dev:  {driver_f1_stdev}\n")
    # MODIFIED BLOCK END

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_RF.py <percentage>")
        sys.exit(1)
    main(sys.argv[1])
