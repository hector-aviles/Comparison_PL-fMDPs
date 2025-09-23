import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle
import time
import statistics
import os
import fnmatch
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main(percentage):
    print("Percentage received:", percentage, flush=True)

    # Convert the percentage to an integer
    try:
        percentage_int = int(percentage)
        print("Percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        sys.exit(1)

    train_path = f"./Train_{percentage}/training_datasets/"
    validation_path = f"./Train_{percentage}/validation_datasets/"
    
    num_train_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print('Train file count:', num_train_files, flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    time_list = []  
    best_params_list = []  # List to store best parameters for each fold

    # Define the hyperparameter grid manually
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 10],
        'max_features': ['sqrt', 'log2']
    }

    # Train each model in the list
    for i in range(1, num_train_files + 1):
        print("Fold:", i, flush=True)    

        # Load training and validation data files
        train_file = f"{train_path}train_fold_{i}.csv"
        validation_file = f"{validation_path}validation_fold_{i}.csv"

        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(validation_file):
            print(f"Missing files for fold {i}. Skipping.", flush=True)
            continue

        # Load train and validation data
        train_data = pd.read_csv(train_file)[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        validation_data = pd.read_csv(validation_file)[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]

        X_train = train_data.drop(['action'], axis=1)
        y_train = encoder.transform(train_data['action'])
        X_val = validation_data.drop(['action'], axis=1)
        y_val = encoder.transform(validation_data['action'])

        print("X_train.head:\n", X_train.head(), flush=True)
        print("X_train.info:")
        X_train.info()
        print(flush=True)

        # Manual hyperparameter evaluation
        best_f1_score = -1
        best_params = {}
        best_model = None

        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for min_samples_split in param_grid['min_samples_split']:
                    #for min_samples_leaf in param_grid['min_samples_leaf']:
                        for max_features in param_grid['max_features']:
                            # Train model with current hyperparameters
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                #min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                random_state=125
                            )

                            start_time = time.time()
                            model.fit(X_train, y_train)
                            end_time = time.time()
                            training_time = end_time - start_time
                            time_list.append(training_time)

                            # Evaluate on validation set
                            y_pred = model.predict(X_val)
                            f1 = f1_score(y_val, y_pred, average='weighted')
                            print(f"Params: n_estimators={n_estimators}, max_depth={max_depth}, "
                                  #f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                                  f"min_samples_split={min_samples_split} "
                                  f"max_features={max_features}, Validation F1-score: {f1:.4f}", flush=True)

                            # Update best parameters if this model is better
                            if f1 > best_f1_score:
                                best_f1_score = f1
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    #'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features,
                                    'f1_score': f1
                                }
                                best_model = model

        best_params_list.append({'fold': i, **best_params})

        print(f"Best parameters for fold {i}: {best_params}", flush=True)
        print(f"Best validation F1-score for fold {i}: {best_f1_score:.4f}", flush=True)

        # Save the best model
        filename = f"./Train_{percentage}/models/RF/RF_{i}.rf"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(best_model, open(filename, 'wb'))

    # Calculate average training time
    if time_list:
        average_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list)
    else:
        average_time = 0
        stdev_time = 0

    # Save training time and best parameters
    numeralia_filename = f"./Train_{percentage}/models/RF/training_numeralia.txt"
    os.makedirs(os.path.dirname(numeralia_filename), exist_ok=True)
    with open(numeralia_filename, "w") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

    # Save the best parameters to a CSV file
    best_params_df = pd.DataFrame(best_params_list)
    best_params_csv = f"./Train_{percentage}/models/RF/best_parameters.csv"
    os.makedirs(os.path.dirname(best_params_csv), exist_ok=True)
    best_params_df.to_csv(best_params_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_RF.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
