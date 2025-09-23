import sys
import pandas as pd
import time
import os
import fnmatch

def main(percentage):
    print("Percentage received:", percentage, flush=True)

    # Convert the percentage to an integer
    try:
        percentage_int = int(percentage)
        print("Percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        return

    # Paths
    ap_path = f"../Train_{percentage}/models/PL-fMDP/"    

    reference_file = "./reference.csv"
    # Check reference file existence
    if not os.path.exists(reference_file):
        print("Reference file does not exist.", flush=True)
        return

    # Read reference data
    reference_data = pd.read_csv(reference_file)

    # Ensure required columns exist in the reference file
    required_columns = ["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    if not all(col in reference_data.columns for col in required_columns):
        print("Reference file is missing required columns.", flush=True)
        return

    # Count files matching the pattern
    csv_files = fnmatch.filter(os.listdir(ap_path), "ap_*.csv")
    print(f"Found {len(csv_files)} CSV files in {ap_path}.", flush=True)

    # Read the original file
    try:
        original_file = "../D.csv"
        original = pd.read_csv(original_file)
    except Exception as e:
        print(f"Error reading file {original_file}: {e}", flush=True)
        return 

    # Convert boolean columns in `original` to 1/0 if necessary
    for col in []:
        if original[col].dtype == bool or original[col].dtype == object:
            print(f"Converting column '{col}' in `original` from boolean or object to integer.", flush=True)
            original[col] = original[col].replace({True: 1, False: 0, "True": 1, "False": 0}).astype(int)

    # Compare headers of `original` and `reference_data`
    if set(original.columns) != set(reference_data.columns):
        print("Headers of original and reference data do not match.", flush=True)
        return

    # Compare headers of `original` and `reference_data`
    if set(original.columns) != set(reference_data.columns):
        print("Headers of original and reference data do not match.", flush=True)
        return

    # Process each `ap_*.csv` file
    for i, csv_file in enumerate(csv_files, start=1):
        csv_path = os.path.join(ap_path, csv_file)
        print(f"Processing ap file {csv_file}...", flush=True)
        # Read the current file
        try:
            ap_data = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}", flush=True)
            continue

        # Ensure the current file has the required columns
        if not all(col in ap_data.columns for col in required_columns):
            print(f"File {csv_file} is missing required columns.", flush=True)
            continue
        
        # Ensure data types are consistent for comparison
        for col in required_columns:
           if original[col].dtype != ap_data[col].dtype:
              print(f"Column '{col}' has mismatched types. Original: {original[col].dtype}, ap_data: {ap_data[col].dtype}", flush=True)
              original[col] = original[col].astype(original[col].dtype)
              print(f"New original[{col}] type: {original[col].dtype}", flush=True)
        

        # Create a copy of the ap data for comparison
        results = ap_data.copy()

        # Count occurrences of rows in `original`
        results["counts"] = ap_data.apply(
            lambda row: original[(original[required_columns] == row[required_columns].values).all(axis=1)].shape[0], 
            axis=1
        )

        # Add the observed `action` to the results
        results["action_ref"] = reference_data["action"]

        # Compare actions
        results["equal"] = results["action"] == results["action_ref"]

        # Organize results by the "equal" column
        results = results.sort_values(by="equal", ascending=True)

        # Print summary of differences for this file
        differences = results["equal"].value_counts().get(False, 0)
        correspondences = results["equal"].value_counts().get(True, 0)
        print(f"File {csv_file}: Differences = {differences}, Correspondences = {correspondences}", flush=True)

        # Save the results to a separate file for this iteration
        output_file = os.path.join(ap_path, f"comparison_results_{i}.csv")
        results.to_csv(output_file, index=False)
        print(f"Comparison results saved to {output_file}.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 comparison.py <percentage>")
        sys.exit(1)

    percentage = sys.argv[1]
    main(percentage)
