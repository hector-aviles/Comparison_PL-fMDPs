import sys
import pandas as pd
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import importlib.util

# -------------------------------------------------
# Dynamic loading of ActionPolicy
# -------------------------------------------------
def load_action_policy(pyfile):
    """Load ActionPolicy class dynamically from a .py file"""
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


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    data_file = "./complete_DB_discrete.csv"
    ap_file = "./Train_full/PL-fMDP/APClassifier_full_0_1_0_0001_lookup_table.py"
    numeralia = "./Train_full/PL-fMDP/Results/testing_numeralia_lookup_table.txt"

    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    if not os.path.exists(data_file):
        print(f"Dataset not found: {data_file}", flush=True)
        sys.exit(1)

    if not os.path.exists(ap_file):
        print(f"AP classifier not found: {ap_file}", flush=True)
        sys.exit(1)

    print("Using dataset:", data_file, flush=True)
    print("Using AP classifier:", ap_file, flush=True)

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------
    data = pd.read_csv(data_file)

    # -------------------------------------------------
    # Subset: keep only latent_collision == False
    # -------------------------------------------------
    if "latent_collision" not in data.columns:
        print("Error: column 'latent_collision' not found in dataset", flush=True)
        sys.exit(1)

    data = data[data["latent_collision"] == False].copy()

    # -------------------------------------------------
    # Class distribution AFTER subsetting
    # -------------------------------------------------
    print("\nClass distribution in testing dataset (after subsetting):", flush=True)

    class_counts = data["action"].value_counts().sort_index()

    # Ensure zero-count classes are shown
    all_classes = [
        "change_to_left",
        "change_to_right",
        "cruise",
        "keep",
        "swerve_left",
        "swerve_right"
    ]

    class_counts = class_counts.reindex(all_classes, fill_value=0)

    for cls, count in class_counts.items():
        print(f"  {cls:16s}: {count}", flush=True)

    print("", flush=True)

    # -------------------------------------------------
    # Keep only required columns (drop latent_collision)
    # -------------------------------------------------
    data = data[
        ["action", "curr_lane",
         "free_E", "free_NE", "free_NW",
         "free_SE", "free_SW", "free_W"]
    ]

    X = data.drop(columns=["action"])
    y_true = data["action"]

    # -------------------------------------------------
    # Label encoding (complete action space)
    # -------------------------------------------------
    encoder = LabelEncoder()
    encoder.classes_ = np.array(all_classes)

    y_true_enc = encoder.transform(y_true)

    # -------------------------------------------------
    # Load policy
    # -------------------------------------------------
    action_policy = load_action_policy(ap_file)
    if action_policy is None:
        sys.exit(1)

    # -------------------------------------------------
    # Prediction + timing
    # -------------------------------------------------
    print("Running inference on non-collision dataset...", flush=True)

    start_time = time.time()
    y_pred = action_policy.predict(X)
    end_time = time.time()

    if y_pred is None:
        print("Error: predict() returned None", flush=True)
        sys.exit(1)

    test_time = end_time - start_time
    y_pred_enc = encoder.transform(y_pred)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    precision = precision_score(y_true_enc, y_pred_enc, average="weighted")
    recall = recall_score(y_true_enc, y_pred_enc, average="weighted")
    f1 = f1_score(y_true_enc, y_pred_enc, average="weighted")
    accuracy = accuracy_score(y_true_enc, y_pred_enc)

    n_rows = len(X)

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    print("Results:", flush=True)
    print(f"  Rows tested (latent_collision == False): {n_rows}", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  F1-score:  {f1:.4f}", flush=True)
    print(f"  Accuracy:  {accuracy:.4f}", flush=True)
    print(f"  Total time: {test_time:.4f}s", flush=True)
    print(f"  Time per row: {test_time / n_rows:.6f}s", flush=True)

    # -------------------------------------------------
    # Save numeralia
    # -------------------------------------------------
    with open(numeralia, "w") as file:
        file.write("PL-fMDP SINGLE-DATASET TEST RESULTS\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Dataset: {data_file}\n")
        file.write("Subset: latent_collision == False\n")
        file.write(f"Model: {ap_file}\n\n")

        file.write("Class distribution (after subsetting):\n")
        for cls, count in class_counts.items():
            file.write(f"  {cls}: {count}\n")
        file.write("\n")

        file.write(f"Rows tested: {n_rows}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Total test time: {test_time:.4f}s\n")
        file.write(f"Time per row: {test_time / n_rows:.6f}s\n")

    print("Testing completed successfully.", flush=True)


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main()

