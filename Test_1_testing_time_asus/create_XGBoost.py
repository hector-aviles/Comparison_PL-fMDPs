import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import pickle
import time
import statistics
import os
import fnmatch
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def main(percentage):
    print("percentage received:", percentage, flush=True)
    # Convert the percentage to an integer
    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        sys.exit(1)

    train_path = f"./Train_{percentage}/training_datasets/"
    validation_path = f"./Train_{percentage}/validation_datasets/"
    models_path = f"./Train_{percentage}/models/XGBoost/"
    os.makedirs(models_path, exist_ok=True)

    num_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print('Train file count:', num_files, flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    time_list = []  
    best_params_list = []  # To store best parameters for each fold

    # Define the hyperparameter grid manually
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'subsample': [1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Train each model in the list
    for i in range(1, num_files + 1):
        print("Fold:", i, flush=True)    

        # Load training data
        train_file = f"{train_path}/train_fold_{i}.csv"
        validation_file = f"{validation_path}/validation_fold_{i}.csv"

        # Check if files exist
        if not os.path.exists(train_file) or not os.path.exists(validation_file):
            print(f"Missing files for fold {i}. Skipping.", flush=True)
            continue

        train_data = pd.read_csv(train_file)
        train_data = train_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]          
        X_train = train_data.drop(['action'], axis=1)
        y_train = encoder.transform(train_data['action'])

        # Load validation data
        validation_data = pd.read_csv(validation_file)
        validation_data = validation_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_val = validation_data.drop(['action'], axis=1)
        y_val = encoder.transform(validation_data['action'])

        print("X_train shape:", X_train.shape, flush=True)
        print("X_val shape:", X_val.shape, flush=True)        

        # Manual hyperparameter evaluation
        best_f1_score = -1
        best_params = {}
        best_model = None

        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for n_estimators in param_grid['n_estimators']:
                    for subsample in param_grid['subsample']:
                        for colsample_bytree in param_grid['colsample_bytree']:
                            # Train model with current hyperparameters
                            model = xgb.XGBClassifier(
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                n_estimators=n_estimators,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                use_label_encoder=False,
                                eval_metric='mlogloss'
                            )

                            start_time = time.time()
                            model.fit(X_train, y_train)
                            end_time = time.time()
                            training_time = end_time - start_time
                            time_list.append(training_time) 

                            # Evaluate on validation set
                            y_pred = model.predict(X_val)
                            f1 = f1_score(y_val, y_pred, average='weighted')
                            print(f"Params: max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, "
                                  f"subsample={subsample}, colsample_bytree={colsample_bytree}, Validation F1-score: {f1:.4f}", flush=True)

                            # Update best parameters if this model is better
                            if f1 > best_f1_score:
                                best_f1_score = f1
                                best_params = {
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'n_estimators': n_estimators,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'f1_score': f1
                                }
                                best_model = model

        best_params_list.append({'fold': i, **best_params})

        print(f"Best parameters for fold {i}: {best_params}", flush=True)
        print(f"Best validation F1-score for fold {i}: {best_f1_score:.4f}", flush=True)

        # Save the best model
        model_filename = f"{models_path}/XGBoost_best_{i}.xgboost"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        pickle.dump(best_model, open(model_filename, 'wb'))

    # Calculate average training time
    if time_list:
        average_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list)
    else:
        average_time = 0
        stdev_time = 0

    # Save training time and best parameters
    numeralia_filename = f"{models_path}/training_numeralia.txt"     
    os.makedirs(os.path.dirname(numeralia_filename), exist_ok=True)
    with open(numeralia_filename, "w") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

    # Save best parameters to a CSV file
    best_params_df = pd.DataFrame(best_params_list)
    best_params_csv = f"{models_path}/best_parameters.csv"
    os.makedirs(os.path.dirname(best_params_csv), exist_ok=True)
    best_params_df.to_csv(best_params_csv, index=False)
    print("Saved best parameters to", best_params_csv, flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_XGBoost.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
