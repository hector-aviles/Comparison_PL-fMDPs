import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import ComplementNB
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
    num_validation_files = len(fnmatch.filter(os.listdir(validation_path), 'validation_fold_*.csv'))

    print('Train file count:', num_train_files, flush=True)
    print('Validation file count:', num_validation_files, flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    time_list = []
    best_params_list = []  # List to store best parameters for each fold

    # Define the hyperparameter grid for ComplementNB
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_prior': [True, False]
    }

    # Train each model in the list
    for i in range(1, num_train_files + 1):
        print("Fold:", i, flush=True)

        # Load training and validation files
        train_file = f"{train_path}train_fold_{i}.csv"
        validation_file = f"{validation_path}validation_fold_{i}.csv"

        if not os.path.exists(train_file) or not os.path.exists(validation_file):
            print(f"Missing files for fold {i}. Skipping.", flush=True)
            continue

        # Load training and validation data
        train_data = pd.read_csv(train_file)
        validation_data = pd.read_csv(validation_file)

        # Select relevant features
        train_data = train_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        validation_data = validation_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]

        X_train = train_data.drop(['action'], axis=1)
        y_train = encoder.transform(train_data['action'])

        X_val = validation_data.drop(['action'], axis=1)
        y_val = encoder.transform(validation_data['action'])

        # Manual hyperparameter evaluation
        best_f1_score = -1
        best_params = {}
        best_model = None

        for alpha in param_grid['alpha']:
            for fit_prior in param_grid['fit_prior']:
                # Train model with current hyperparameters
                nb = ComplementNB(alpha=alpha, fit_prior=fit_prior)
                start_time = time.time()
                nb.fit(X_train, y_train)
                end_time = time.time()
                training_time = end_time - start_time
                time_list.append(training_time)

                # Evaluate on validation set
                y_pred = nb.predict(X_val)
                f1 = f1_score(y_val, y_pred, average='weighted')
                print(f"Params: alpha={alpha}, fit_prior={fit_prior}, Validation F1-score: {f1:.4f}", flush=True)

                # Update best parameters if this model is better
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_params = {'alpha': alpha, 'fit_prior': fit_prior, 'f1_score': f1}
                    best_model = nb

        best_params_list.append({'fold': i, **best_params})

        print(f"Best parameters for fold {i}: {best_params}", flush=True)
        print(f"Best validation F1-score for fold {i}: {best_f1_score:.4f}", flush=True)

        # Save the best model
        model_filename = f"./Train_{percentage}/models/NB/NB_{i}.nb"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        pickle.dump(best_model, open(model_filename, 'wb'))

    # Calculate average and standard deviation of training times
    if time_list:
        average_time = statistics.mean(time_list)
        stdev_time = statistics.stdev(time_list)
    else:
        average_time = 0
        stdev_time = 0

    numeralia_filename = f"./Train_{percentage}/models/NB/training_numeralia.txt"
    os.makedirs(os.path.dirname(numeralia_filename), exist_ok=True)
    with open(numeralia_filename, "w") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")

    # Save the best parameters to a CSV file
    best_params_df = pd.DataFrame(best_params_list)
    best_params_csv = f"./Train_{percentage}/models/NB/best_parameters.csv"
    os.makedirs(os.path.dirname(best_params_csv), exist_ok=True)
    best_params_df.to_csv(best_params_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_NB.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
