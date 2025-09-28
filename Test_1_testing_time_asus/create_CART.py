import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pickle
import time
import statistics
import os
import fnmatch
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main(percentage):
    print("percentage received:", percentage, flush=True)
    # Convert the percentage to an integer
    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        return
        
    train_path = "./Train_" + percentage + "/training_datasets/"
    validation_path = "./Train_" + percentage + "/validation_datasets/"
    models_path = "./Train_" + percentage + "/models/CART/"
    os.makedirs(models_path, exist_ok=True)
    
    num_files = len(fnmatch.filter(os.listdir(train_path), 'train_fold_*.csv'))
    print('Train file count:', num_files, flush=True)
    
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])
    
    time_list = []
    best_params_list = []
    
    # Hyperparameter grid
    param_grid = {
        "max_depth": [2, 4, 6],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10]
    }
    
    # Train each model in the list
    for i in range(1, num_files + 1):
        print(f"Fold: {i}", flush=True)
        
        # Load training data
        train_file = f"{train_path}/train_fold_{i}.csv"
        train_data = pd.read_csv(train_file)
        train_data = train_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_train = train_data.drop(['action'], axis=1)
        y_train = encoder.transform(train_data['action'])
        
        # Load validation data
        validation_file = f"{validation_path}/validation_fold_{i}.csv"
        validation_data = pd.read_csv(validation_file)
        validation_data = validation_data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        X_val = validation_data.drop(['action'], axis=1)
        y_val = encoder.transform(validation_data['action'])
        
        print("X_train shape:", X_train.shape, flush=True)
        print("X_val shape:", X_val.shape, flush=True)
        
        best_model = None
        best_params = None
        best_f1_score = -1
        
        # Grid search for hyperparameters
        for max_depth in param_grid["max_depth"]:
            for min_samples_split in param_grid["min_samples_split"]:
                for min_samples_leaf in param_grid["min_samples_leaf"]:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf
                    )
                    
                    # Measure training time
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    end_time = time.time()
                    
                    # Record training time
                    training_time = end_time - start_time
                    time_list.append(training_time)
                    
                    # Validation F1-score
                    y_pred = model.predict(X_val)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    
                    # Check if this model is better
                    if f1 > best_f1_score:
                        best_f1_score = f1
                        best_model = model
                        best_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "f1_score": f1
                        }
        
        # Save the best model for this fold
        model_filename = f"{models_path}/CART_{i}.cart"
        pickle.dump(best_model, open(model_filename, 'wb'))
        
        # Save best parameters for this fold
        best_params_list.append({"fold": i, **best_params})
        
        print(f"Best parameters for fold {i}: {best_params}", flush=True)
        print(f"Best validation F1-score for fold {i}: {best_f1_score:.4f}", flush=True)
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.to_csv(f"{models_path}/best_parameters.csv", index=False)
    
    # Calculate average training time
    average_time = statistics.mean(time_list)
    stdev_time = statistics.stdev(time_list)
    
    # Write training numeralia
    numeralia_filename = f"{models_path}/training_numeralia.txt"
    with open(numeralia_filename, "w") as file:
        file.write(f"Avg. training time: {average_time:.4f}\n")
        file.write(f"Stdev. training time: {stdev_time:.4f}\n")
    
    print("All models trained. Best parameters saved.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 create_CART.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
