import sys
import pandas as pd
from sklearn.metrics import f1_score
import pickle
import os
import fnmatch
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import statistics

def main(percentage):
    print("Percentage received:", percentage, flush=True)
    try:
        percentage_int = int(percentage)
        print("Percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        sys.exit(1)

    validation_path = f"./Train_{percentage}/validation_datasets/"
    train_path = f"./Train_{percentage}/training_datasets/"
    
    num_validation_files = len(fnmatch.filter(os.listdir(validation_path), 'validation_fold_*.csv'))
    num_train_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print('Train file count:', num_train_files, flush=True)
    print('Validation file count:', num_validation_files, flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    # Hyperparameter options
    hyperparameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'class_weight': [None, 'balanced']
    }
    best_hyperparameters = []

    output_file = f"./Train_{percentage}/models/LR/best_parameters.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write("fold,C,penalty,class_weight,F1_Score\n")  # Updated header

    time_list = []
    for i in range(1, num_validation_files + 1):
        print(f"Manual hyperparameter tuning for Fold: {i}", flush=True)
        validation_file = f"{validation_path}validation_fold_{i}.csv"
        train_file = f"{train_path}train_fold_{i}.csv"

        if not os.path.exists(train_file) or not os.path.exists(validation_file):
            print(f"Missing files for fold {i}. Skipping.", flush=True)
            continue

        # Load validation and training data
        validation_data = pd.read_csv(validation_file)
        validation_data = validation_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        train_data = pd.read_csv(train_file)
        train_data = train_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]

        X_val = validation_data.drop(columns=['action'])
        y_val = encoder.transform(validation_data['action'])
        X_train = train_data.drop(columns=['action'])
        y_train = encoder.transform(train_data['action'])

        # Manual hyperparameter tuning
        best_f1 = -1
        best_params = None
        best_model = None

        for C in hyperparameters['C']:
            for penalty in hyperparameters['penalty']:
                for class_weight in hyperparameters['class_weight']:
                    print(f"Evaluating C={C}, penalty={penalty}, class_weight={class_weight}...", flush=True)
                    try:
                        # Train the model
                        model = LogisticRegression(C=C, penalty=penalty, class_weight=class_weight, solver='liblinear', max_iter=500)
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        end_time = time.time()
                        
                        # Record training time
                        training_time = end_time - start_time
                        time_list.append(training_time)

                        # Validate the model
                        y_pred = model.predict(X_val)
                        f1 = f1_score(y_val, y_pred, average='weighted')

                        print(f"F1 Score for C={C}, penalty={penalty}, class_weight={class_weight}: {f1:.4f}", flush=True)

                        # Update best parameters
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {'C': C, 'penalty': penalty, 'class_weight': class_weight, 'f1_score': f1}
                            best_model = model

                    except Exception as e:
                        print(f"Error with C={C}, penalty={penalty}, class_weight={class_weight}: {e}", flush=True)
                        continue

        if best_params:
            best_hyperparameters.append(best_params)
            print(f"Best hyperparameters for Fold {i}: {best_params}", flush=True)

            # Append best parameters to the CSV file
            with open(output_file, 'a') as file:
                file.write(f"{i},{best_params['C']},{best_params['penalty']},{best_params['class_weight']},{best_params['f1_score']:.4f}\n")
                
            # Save the best model
            model_filename = f"./Train_{percentage}/models/LR/LR_{i}.lr"
            os.makedirs(os.path.dirname(model_filename), exist_ok=True)
            pickle.dump(best_model, open(model_filename, 'wb'))
        else:
            print(f"No valid hyperparameters found for Fold {i}", flush=True)

    # Save average training time
    if time_list:
        average_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list) if len(time_list) > 1 else 0
    else:
        average_time = 0
        stdev_time = 0

    numeralia_filename = f"./Train_{percentage}/models/LR/training_numeralia.txt"
    os.makedirs(os.path.dirname(numeralia_filename), exist_ok=True)
    with open(numeralia_filename, "w") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_LR.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
