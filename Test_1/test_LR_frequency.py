import sys
import pandas as pd
import os
import re
import time
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
import pickle

def load_lr_model(model_file):
    """Load a fitted LR (Logistic Regression) model from .lr / .pkl / .joblib file"""
    try:
        model = joblib.load(model_file)
        print(f"Loaded LR model from {model_file}", flush=True)
        return model
    except Exception as e:
        print(f"Failed to load {model_file}: {str(e)}", flush=True)
        return None

def get_frequency_range(count):
    if count <= 4:
        return "Low (0-4)"
    elif count <= 100:
        return "Medium (5-100)"
    elif count <= 2000:
        return "Medium-High (101-2000)"
    else:
        return "High (>2000)"

def extract_fold_number(filename):
    """Extract fold number from filenames like LR_5.lr, LR_fold5.lr, logistic_3.lr, etc."""
    match = re.search(r'(?i)(?:LR|logistic|logreg)[_fold]*(\d+)\.(lr|pkl|joblib)', filename)
    if match:
        return int(match.group(1))
    return None

# ────────────────────────────────────────────────────────────────
# Safety functions (same as PL-fMDP)
# ────────────────────────────────────────────────────────────────
def create_safe_lookup(safe_file):
    safe_df = pd.read_csv(safe_file)
    example_cols = ['action', 'curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    bool_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    for col in bool_cols:
        safe_df[col] = safe_df[col].astype(str).map({
            '1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
            '1.0': 1, '0.0': 0, '1.': 1, '0.': 0
        }).fillna(0).astype(int)
    return set(tuple(row) for row in safe_df[example_cols].itertuples(index=False, name=None))

def check_action_safety(action, state_row, feature_cols, safe_lookup):
    state_values = [state_row[col] for col in feature_cols]
    state_action = tuple([action] + state_values)
    return state_action in safe_lookup

def get_frequency_category(count):
    if count == 0:
        return "Zero (0)"
    elif count <= 4:
        return "Very Low (1-4)"
    elif count <= 100:
        return "Low (5-100)"
    elif count <= 2000:
        return "Medium (101-2000)"
    else:
        return "High (>2000)"

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Using training percentage: {percentage_int}%", flush=True)
    except ValueError:
        print("Percentage must be an integer.", flush=True)
        sys.exit(1)

    # Paths — now pointing to LR folder
    lr_path = f"./Train_{percentage}/models/LR/"
    analysis_file = "./analysis/count_sample_space_auto_with_safe.csv"
    safe_file = "./analysis/no_crashes.csv"
    numeralia = os.path.join(lr_path, "Results", "testing_lr_frequency.txt")
    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    if not os.path.exists(analysis_file):
        print(f"Analysis file not found: {analysis_file}", flush=True)
        sys.exit(1)

    full_data = pd.read_csv(analysis_file)
    print(f"Loaded {len(full_data)} samples from {analysis_file}", flush=True)

    feature_columns = ["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    target_col = "action"
    count_col = "count"

    missing = [c for c in feature_columns + [target_col, count_col] if c not in full_data.columns]
    if missing:
        print(f"Missing required columns: {missing}", flush=True)
        sys.exit(1)

    full_data = full_data[feature_columns + [target_col, count_col]].copy()

    for col in feature_columns:
        full_data[col] = full_data[col].astype(str).map({
            '1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
            '1.0': 1, '0.0': 0, '1.': 1, '0.': 0
        }).fillna(0).astype(int)

    full_data['frequency_range'] = full_data[count_col].apply(get_frequency_range)
    frequency_ranges = sorted(full_data['frequency_range'].unique())

    # Finer bins for safety
    full_data['frequency_category'] = full_data[count_col].apply(get_frequency_category)
    freq_categories = sorted(full_data['frequency_category'].unique())

    safe_lookup = create_safe_lookup(safe_file)
    print(f"Safe state-action pairs: {len(safe_lookup):,}", flush=True)

    known_classes = ['change_to_left', 'change_to_right', 'cruise', 'keep']
    encoder = LabelEncoder()
    encoder.classes_ = np.array(known_classes)

    # Model discovery
    all_files = os.listdir(lr_path)
    model_files = []
    for filename in all_files:
        if filename.lower().endswith('.lr'):
            fold_num = extract_fold_number(filename)
            if fold_num is not None:
                model_files.append((fold_num, filename))
    model_files.sort()

    if len(model_files) == 0:
        print(f"No .lr files found in {lr_path}", flush=True)
        sys.exit(1)

    print(f"Found {len(model_files)} LR files (folds: {[n for n,_ in model_files]})", flush=True)

    # Performance results
    results = {fr: {'precisions': [], 'recalls': [], 'f1_scores': [],
                    'accuracies': [], 'test_times': [], 'test_sizes': [],
                    'fold_numbers': []} for fr in frequency_ranges}

    # Safety disagreement aggregators
    disagreement_safety = {
        cat: {
            'total_disagreements': 0,
            'total_samples': 0,
            'dsafe_lr_safe': 0,
            'dsafe_lr_unsafe': 0,
            'dunsafe_lr_safe': 0,
            'dunsafe_lr_unsafe': 0,
            'by_action': {a: {'total':0, 'dsafe_lr_safe':0, 'dsafe_lr_unsafe':0,
                              'dunsafe_lr_safe':0, 'dunsafe_lr_unsafe':0}
                          for a in known_classes}
        }
        for cat in freq_categories
    }
    global_disagreement = {'total_disagreements': 0,
                           'dsafe_lr_safe': 0, 'dsafe_lr_unsafe': 0,
                           'dunsafe_lr_safe': 0, 'dunsafe_lr_unsafe': 0}

    # Write header
    with open(numeralia, "w", encoding="utf-8") as file:
        file.write("Logistic Regression (LR) Testing Results by Frequency Range\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"Source file: {analysis_file}\n")
        file.write(f"Safe examples source: {safe_file}\n")
        file.write(f"Total rows tested: {len(full_data)}\n")
        file.write(f"Training percentage: {percentage}%\n\n")
        file.write("NOTE: All rows in the CSV are evaluated (one per unique state-action combo).\n")
        file.write("'count' is used only to assign frequency ranges.\n\n")
        file.write(f"Input features: {', '.join(feature_columns)}\n")
        file.write("Output: action\n\n")
        file.write("Frequency Range Distribution (based on 'count' column):\n")
        for fr in sorted(frequency_ranges):
            count = len(full_data[full_data['frequency_range'] == fr])
            file.write(f" {fr}: {count:,} samples\n")
        file.write("\n" + "=" * 80 + "\n\n")

    # Main loop
    for fold_num, model_filename in model_files:
        print(f"\n{'='*70}", flush=True)
        print(f"Processing Fold {fold_num} → {model_filename}", flush=True)

        model_file = os.path.join(lr_path, model_filename)
        lr_model = load_lr_model(model_file)
        if lr_model is None:
            continue

        fold_has_data = False

        for frequency_range in frequency_ranges:
            range_data = full_data[full_data['frequency_range'] == frequency_range]
            if len(range_data) == 0:
                print(f" No samples for {frequency_range}, skipping", flush=True)
                continue

            X_test = range_data[feature_columns].values.astype(float)
            y_test = encoder.transform(range_data[target_col])
            test_size = len(X_test)

            try:
                start_time = time.time()
                y_pred = lr_model.predict(X_test)
                end_time = time.time()
                y_pred_encoded = y_pred.astype(int)  # LR returns class indices
                test_time = end_time - start_time

                precision = precision_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred_encoded, average='weighted', zero_division=0)
                accuracy = accuracy_score(y_test, y_pred_encoded)

                results[frequency_range]['precisions'].append(precision)
                results[frequency_range]['recalls'].append(recall)
                results[frequency_range]['f1_scores'].append(f1)
                results[frequency_range]['accuracies'].append(accuracy)
                results[frequency_range]['test_times'].append(test_time)
                results[frequency_range]['test_sizes'].append(test_size)
                results[frequency_range]['fold_numbers'].append(fold_num)

                fold_has_data = True

                print(f" {frequency_range}: Size: {test_size:,}, Prec: {precision:.4f}, Rec: {recall:.4f}, "
                      f"F1: {f1:.4f}, Acc: {accuracy:.4f}, Time: {test_time:.4f}s", flush=True)

                with open(numeralia, "a", encoding="utf-8") as file:
                    file.write(f"\nFold {fold_num} - {frequency_range}:\n")
                    file.write("-" * 40 + "\n")
                    file.write(f"Test size: {test_size:,} rows\n")
                    file.write(f"Precision: {precision:.4f}\n")
                    file.write(f"Recall: {recall:.4f}\n")
                    file.write(f"F1-score: {f1:.4f}\n")
                    file.write(f"Accuracy: {accuracy:.4f}\n")
                    file.write(f"Test time: {test_time:.4f}s\n")
                    file.write(f"Time per row: {test_time/test_size:.6f}s\n\n")

            except Exception as e:
                print(f" Error testing {frequency_range} in fold {fold_num}: {str(e)}", flush=True)
                continue

        # ── Safety & disagreement analysis ────────────────────────────────
        y_pred_all = lr_model.predict(full_data[feature_columns].values.astype(float))
        for i, idx in enumerate(full_data.index):
            row = full_data.loc[idx]
            true_action = row[target_col]
            pred_action = encoder.inverse_transform([int(y_pred_all[i])])[0]

            if true_action == pred_action:
                continue

            cat = row['frequency_category']
            state_row = row[feature_columns]

            ds_safe = check_action_safety(true_action, state_row, feature_columns, safe_lookup)
            lr_safe = check_action_safety(pred_action, state_row, feature_columns, safe_lookup)

            disagreement_safety[cat]['total_disagreements'] += 1
            disagreement_safety[cat]['total_samples'] += 1
            disagreement_safety[cat]['by_action'][true_action]['total'] += 1

            global_disagreement['total_disagreements'] += 1

            if ds_safe and lr_safe:
                k = 'dsafe_lr_safe'
            elif ds_safe and not lr_safe:
                k = 'dsafe_lr_unsafe'
            elif not ds_safe and lr_safe:
                k = 'dunsafe_lr_safe'
            else:
                k = 'dunsafe_lr_unsafe'

            disagreement_safety[cat][k] += 1
            disagreement_safety[cat]['by_action'][true_action][k] += 1
            global_disagreement[k] += 1

        if fold_has_data:
            print(f" Fold {fold_num} completed with data", flush=True)

    # ── Original aggregated performance ──────────────────────────────────────
    with open(numeralia, "a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write("AGGREGATED RESULTS PER FREQUENCY RANGE (mean ± std across folds)\n")
        file.write("=" * 80 + "\n\n")

        for frequency_range in sorted(frequency_ranges):
            range_results = results[frequency_range]
            if range_results['test_times']:
                n_tests = len(range_results['test_times'])
                avg_size = statistics.mean(range_results['test_sizes'])
                total_samples = sum(range_results['test_sizes'])
                fold_numbers = range_results['fold_numbers']

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

                file.write(f"{frequency_range}:\n")
                file.write("-" * 40 + "\n")
                file.write(f" Valid folds: {n_tests}/20 (folds: {sorted(fold_numbers)})\n")
                file.write(f" Total samples tested: {total_samples:,}\n")
                file.write(f" Average test size per fold: {avg_size:,.1f}\n")
                file.write(f" Average test time: {metrics['time'][0]:.4f}s ± {metrics['time'][1]:.4f}s\n")
                file.write(f" Average time per row: {metrics['time'][0]/avg_size:.6f}s\n")
                file.write(f" F1-score: {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}\n")
                file.write(f" Precision: {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}\n")
                file.write(f" Recall: {metrics['recall'][0]:.4f} ± {metrics['recall'][1]:.4f}\n")
                file.write(f" Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}\n")
                file.write("-" * 60 + "\n\n")

        # Global summary
        file.write("\n" + "=" * 80 + "\n")
        file.write("GLOBAL SUMMARY ACROSS ALL FREQUENCY RANGES AND FOLDS\n")
        file.write("=" * 80 + "\n\n")
        global_prec = []
        global_rec = []
        global_f1 = []
        global_acc = []
        global_time = []
        global_size = []
        for fr in results:
            global_prec.extend(results[fr]['precisions'])
            global_rec.extend(results[fr]['recalls'])
            global_f1.extend(results[fr]['f1_scores'])
            global_acc.extend(results[fr]['accuracies'])
            global_time.extend(results[fr]['test_times'])
            global_size.extend(results[fr]['test_sizes'])
        n_global = len(global_f1)
        if n_global > 0:
            g_mean_f1 = statistics.mean(global_f1)
            g_std_f1 = statistics.stdev(global_f1) if n_global > 1 else 0.0
            g_mean_prec = statistics.mean(global_prec)
            g_std_prec = statistics.stdev(global_prec) if n_global > 1 else 0.0
            g_mean_rec = statistics.mean(global_rec)
            g_std_rec = statistics.stdev(global_rec) if n_global > 1 else 0.0
            g_mean_acc = statistics.mean(global_acc)
            g_std_acc = statistics.stdev(global_acc) if n_global > 1 else 0.0
            g_mean_time = statistics.mean(global_time)
            g_std_time = statistics.stdev(global_time) if n_global > 1 else 0.0
            g_total_samples = sum(global_size)
            g_avg_size = statistics.mean(global_size)
            g_mean_tpr = g_mean_time / g_avg_size if g_avg_size > 0 else 0
            file.write(f"Total valid predictions: {n_global:,}\n")
            file.write(f"Total samples tested: {g_total_samples:,}\n")
            file.write(f"Avg test size per fold (approx): {g_avg_size:,.1f}\n")
            file.write(f"Avg test time per fold: {g_mean_time:.4f}s ± {g_std_time:.4f}s\n")
            file.write(f"Avg time per row: {g_mean_tpr:.6f}s\n")
            file.write(f"Global F1-score: {g_mean_f1:.4f} ± {g_std_f1:.4f}\n")
            file.write(f"Global Precision: {g_mean_prec:.4f} ± {g_std_prec:.4f}\n")
            file.write(f"Global Recall: {g_mean_rec:.4f} ± {g_std_rec:.4f}\n")
            file.write(f"Global Accuracy: {g_mean_acc:.4f} ± {g_std_acc:.4f}\n")
        else:
            file.write("No valid global results\n")

    # ── Safety & disagreement analysis ───────────────────────────────────────
    with open(numeralia, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write("SAFETY & DISAGREEMENT ANALYSIS (finer bins)\n")
        f.write("=" * 90 + "\n\n")

        for cat in sorted(freq_categories):
            s = disagreement_safety[cat]
            td = s['total_disagreements']
            ts = s['total_samples']

            f.write(f"{cat}:\n")
            f.write(f" Total samples: {ts:,}\n")
            f.write(f" Total disagreements: {td:,} ({td/ts*100:.1f}%)\n" if ts > 0 else " Total disagreements: 0\n")

            if td == 0:
                f.write("\n")
                continue

            net = s['dunsafe_lr_safe'] - s['dsafe_lr_unsafe']
            f.write(f" LR Safer: {s['dunsafe_lr_safe']:,} ({s['dunsafe_lr_safe']/td*100:.1f}%)\n")
            f.write(f" LR Riskier: {s['dsafe_lr_unsafe']:,} ({s['dsafe_lr_unsafe']/td*100:.1f}%)\n")
            f.write(f" Net safety improvement: {net:+d}\n")

            if cat in ["Zero (0)", "Very Low (1-4)"]:
                f.write("\nSafety Confusion Matrix:\n")
                f.write(f"Dataset Safe   LR Safe: {s['dsafe_lr_safe']:,}   LR Unsafe: {s['dsafe_lr_unsafe']:,}\n")
                f.write(f"Dataset Unsafe LR Safe: {s['dunsafe_lr_safe']:,}   LR Unsafe: {s['dunsafe_lr_unsafe']:,}\n\n")

                f.write("By dataset action:\n")
                for a in known_classes:
                    st = s['by_action'][a]
                    if st['total'] > 0:
                        f.write(f"  {a}: total {st['total']:,}   Safer {st['dunsafe_lr_safe']:,}   Riskier {st['dsafe_lr_unsafe']:,}\n")
            f.write("\n")

        # Global safety
        f.write("=" * 90 + "\n")
        f.write("GLOBAL SAFETY SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        gt = global_disagreement['total_disagreements']
        if gt > 0:
            net_g = global_disagreement['dunsafe_lr_safe'] - global_disagreement['dsafe_lr_unsafe']
            f.write(f"Total disagreements: {gt:,}\n")
            f.write(f"LR Safer: {global_disagreement['dunsafe_lr_safe']:,} ({global_disagreement['dunsafe_lr_safe']/gt*100:.1f}%)\n")
            f.write(f"LR Riskier: {global_disagreement['dsafe_lr_unsafe']:,} ({global_disagreement['dsafe_lr_unsafe']/gt*100:.1f}%)\n")
            f.write(f"Net: {net_g:+d}\n")
        else:
            f.write("No disagreements found.\n")

    print(f"\nTesting completed. Results saved to: {numeralia}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_lr_frequency.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
