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
import re

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

def get_frequency_range(count):
    """Categorize frequency count into ranges"""
    if count <= 4:
        return "Low (0-4)"
    elif count <= 100:
        return "Medium (5-100)"
    elif count <= 2000:
        return "Medium-High (101-2000)"
    else:
        return "High (>2000)"

def extract_fold_number(filename):
    """Extract the fold number from the filename"""
    match = re.search(r'APClassifier_(\d+)_lookup_table\.py', filename)
    if match:
        return int(match.group(1))
    return None

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Percentage as an integer: {percentage_int}", flush=True)
    except ValueError:
        print("Percentage is not a valid integer.", flush=True)
        sys.exit(1)

    # Paths
    ap_path = f"./Train_{percentage}/models/PL-fMDP/"
    analysis_file = "./analysis/count_sample_space_auto.csv"
    numeralia = f"./Train_{percentage}/models/PL-fMDP/Results/testing_numeralia_lookup_table_frequency.txt"
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    # Load the analysis file with frequency counts
    if not os.path.exists(analysis_file):
        print(f"Analysis file {analysis_file} does not exist!", flush=True)
        sys.exit(1)
    
    full_data = pd.read_csv(analysis_file)
    print(f"Loaded {len(full_data)} samples from {analysis_file}", flush=True)
    
    # Keep only the needed columns (action to free_W)
    feature_columns = ["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    full_data = full_data[feature_columns + ["count"]]  # Keep count for grouping
    
    # Add frequency range column
    full_data['frequency_range'] = full_data['count'].apply(get_frequency_range)
    
    # Get unique frequency ranges
    frequency_ranges = full_data['frequency_range'].unique()
    print(f"Frequency ranges found: {frequency_ranges}", flush=True)

    # Find all model files and extract their fold numbers
    all_model_files = fnmatch.filter(os.listdir(ap_path), 'APClassifier_*_lookup_table.py')
    
    # Create a list of (fold_number, filename) tuples and sort by fold number
    model_files = []
    for filename in all_model_files:
        fold_num = extract_fold_number(filename)
        if fold_num is not None:
            model_files.append((fold_num, filename))
    
    # Sort by fold number
    model_files.sort()
    
    num_files = len(model_files)
    print(f"Found {num_files} model files (from {model_files[0][0]} to {model_files[-1][0]})", flush=True)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])

    # Initialize results storage
    # Structure: results[frequency_range][fold] = {'precision': ..., 'recall': ..., etc.}
    results = {fr: {'precisions': [], 'recalls': [], 'f1_scores': [], 
                    'accuracies': [], 'test_times': [], 'test_sizes': [],
                    'fold_numbers': []}  # Track which folds contributed
               for fr in frequency_ranges}
    
    # Also store per-fold detailed results
    all_fold_results = []

    with open(numeralia, "w") as file:
        file.write("PL-fMDP Testing Results by Frequency Range:\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Source file: {analysis_file}\n")
        file.write(f"Total samples: {len(full_data)}\n\n")
        
        # Write frequency range distribution
        file.write("Frequency Range Distribution:\n")
        for fr in sorted(frequency_ranges):
            count = len(full_data[full_data['frequency_range'] == fr])
            file.write(f"  {fr}: {count} samples\n")
        file.write("\n" + "=" * 60 + "\n\n")

    # Process each model file
    for fold_num, model_filename in model_files:
        print(f"\n{'='*50}", flush=True)
        print(f"Processing Fold {fold_num}", flush=True)
        print(f"{'='*50}", flush=True)
        
        pyfile = os.path.join(ap_path, model_filename)
        
        if not os.path.exists(pyfile):
            print(f"Model file {pyfile} does not exist, skipping fold {fold_num}", flush=True)
            continue

        # Load the model
        action_policy_instance = load_action_policy(pyfile)
        if action_policy_instance is None:
            continue

        # Test on each frequency range
        fold_results = {'fold': fold_num}
        fold_has_data = False
        
        for frequency_range in frequency_ranges:
            # Get samples for this frequency range
            range_data = full_data[full_data['frequency_range'] == frequency_range]
            
            if len(range_data) == 0:
                print(f"  No samples for {frequency_range}, skipping", flush=True)
                continue
            
            # Prepare test data
            X_test = range_data[feature_columns[1:]]  # Exclude 'action' column
            y_test = encoder.transform(range_data['action'])
            
            test_size = len(X_test)
            
            try:
                # Predict
                start_time = time.time()
                y_pred = action_policy_instance.predict(X_test)
                end_time = time.time()
                
                if y_pred is None:
                    print(f"  Error: predict returned None for {frequency_range}", flush=True)
                    continue
                
                test_time = end_time - start_time
                y_pred_encoded = encoder.transform(y_pred)
                
                # Compute metrics
                precision = precision_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_encoded)
                
                # Store results
                results[frequency_range]['precisions'].append(precision)
                results[frequency_range]['recalls'].append(recall)
                results[frequency_range]['f1_scores'].append(f1)
                results[frequency_range]['accuracies'].append(accuracy)
                results[frequency_range]['test_times'].append(test_time)
                results[frequency_range]['test_sizes'].append(test_size)
                results[frequency_range]['fold_numbers'].append(fold_num)
                
                # Store in fold results
                fold_results[f'{frequency_range}_size'] = test_size
                fold_results[f'{frequency_range}_precision'] = precision
                fold_results[f'{frequency_range}_recall'] = recall
                fold_results[f'{frequency_range}_f1'] = f1
                fold_results[f'{frequency_range}_accuracy'] = accuracy
                fold_results[f'{frequency_range}_time'] = test_time
                
                fold_has_data = True
                
                print(f"  {frequency_range}:", flush=True)
                print(f"    Size: {test_size}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                      f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Time: {test_time:.4f}s", flush=True)
                
            except Exception as e:
                print(f"  Error testing {frequency_range} in fold {fold_num}: {str(e)}", flush=True)
                continue
        
        # Store fold results if we have data
        if fold_has_data:
            all_fold_results.append(fold_results)
            
            # Write detailed fold results
            with open(numeralia, "a") as file:
                file.write(f"\nFold {fold_num} Results:\n")
                file.write("-" * 40 + "\n")
                for frequency_range in sorted(frequency_ranges):
                    if f'{frequency_range}_size' in fold_results:
                        time_per_row = fold_results[f'{frequency_range}_time'] / fold_results[f'{frequency_range}_size']
                        file.write(f"{frequency_range}:\n")
                        file.write(f"  Test size: {fold_results[f'{frequency_range}_size']} rows\n")
                        file.write(f"  Precision: {fold_results[f'{frequency_range}_precision']:.4f}\n")
                        file.write(f"  Recall: {fold_results[f'{frequency_range}_recall']:.4f}\n")
                        file.write(f"  F1-score: {fold_results[f'{frequency_range}_f1']:.4f}\n")
                        file.write(f"  Accuracy: {fold_results[f'{frequency_range}_accuracy']:.4f}\n")
                        file.write(f"  Test time: {fold_results[f'{frequency_range}_time']:.4f}s\n")
                        file.write(f"  Time per row: {time_per_row:.6f}s\n\n")

    # Calculate and write overall statistics
    with open(numeralia, "a") as file:
        file.write("\n" + "=" * 60 + "\n")
        file.write("OVERALL TESTING STATISTICS BY FREQUENCY RANGE\n")
        file.write("=" * 60 + "\n\n")
        
        for frequency_range in sorted(frequency_ranges):
            range_results = results[frequency_range]
            
            if range_results['test_times']:
                n_tests = len(range_results['test_times'])
                avg_size = statistics.mean(range_results['test_sizes'])
                total_samples = sum(range_results['test_sizes'])
                fold_numbers = range_results['fold_numbers']
                
                # Calculate statistics
                metrics = {
                    'precision': (statistics.mean(range_results['precisions']), 
                                 statistics.stdev(range_results['precisions']) if n_tests > 1 else 0),
                    'recall': (statistics.mean(range_results['recalls']), 
                              statistics.stdev(range_results['recalls']) if n_tests > 1 else 0),
                    'f1': (statistics.mean(range_results['f1_scores']), 
                          statistics.stdev(range_results['f1_scores']) if n_tests > 1 else 0),
                    'accuracy': (statistics.mean(range_results['accuracies']), 
                                statistics.stdev(range_results['accuracies']) if n_tests > 1 else 0),
                    'time': (statistics.mean(range_results['test_times']), 
                            statistics.stdev(range_results['test_times']) if n_tests > 1 else 0)
                }
                
                file.write(f"\n{frequency_range}:\n")
                file.write("-" * 40 + "\n")
                file.write(f"  Folds tested: {n_tests}\n")
                file.write(f"  Fold numbers: {sorted(fold_numbers)}\n")
                file.write(f"  Total samples tested: {total_samples}\n")
                file.write(f"  Average test size per fold: {avg_size:.1f} rows\n")
                file.write(f"  Average test time: {metrics['time'][0]:.4f}s ± {metrics['time'][1]:.4f}s\n")
                file.write(f"  Average time per row: {metrics['time'][0]/avg_size:.6f}s\n")
                file.write(f"  Average F1-score: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}\n")
                file.write(f"  Average Precision: {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}\n")
                file.write(f"  Average Recall: {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}\n")
                file.write(f"  Average Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}\n")
                
                # Also print to console
                print(f"\n{frequency_range} Summary:", flush=True)
                print(f"  Folds tested: {n_folds} (folds {sorted(fold_numbers)[:3]}...)" if n_folds > 3 else f"  Folds tested: {n_folds} (folds {sorted(fold_numbers)})", flush=True)
                print(f"  Total samples: {total_samples}", flush=True)
                print(f"  Avg F1: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}", flush=True)
                print(f"  Avg Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}", flush=True)
            else:
                file.write(f"\n{frequency_range}: No results available\n")
    
    # Write summary of all folds processed
    with open(numeralia, "a") as file:
        file.write("\n" + "=" * 60 + "\n")
        file.write("SUMMARY OF PROCESSED FOLDS\n")
        file.write("=" * 60 + "\n")
        file.write(f"Total model files found: {num_files}\n")
        file.write(f"Folds successfully processed: {len(all_fold_results)}\n")
        if all_fold_results:
            fold_numbers_processed = [r['fold'] for r in all_fold_results]
            file.write(f"Fold numbers processed: {sorted(fold_numbers_processed)}\n")
    
    print(f"\nTesting completed. Results saved to: {numeralia}", flush=True)
    print(f"Processed {len(all_fold_results)} out of {num_files} model files", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_APs_lookup_table.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
