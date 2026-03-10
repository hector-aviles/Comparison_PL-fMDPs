import sys
import pandas as pd
import os
import re
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import statistics
import importlib.util
from collections import Counter

def load_action_policy(pyfile):
    try:
        spec = importlib.util.spec_from_file_location("ap_module", pyfile)
        if spec is None:
            print(f"Cannot create spec for {pyfile}", flush=True)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "ActionPolicy"):
            print(f"No 'ActionPolicy' class found in {pyfile}", flush=True)
            return None
        return module.ActionPolicy()
    except Exception as e:
        print(f"Failed to load {pyfile}: {str(e)}", flush=True)
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

def extract_fold_number(filename):
    match = re.search(r'APClassifier_(\d+)_lookup_table\.py', filename)
    if match:
        return int(match.group(1))
    return None

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

def main(percentage):
    print(f"Percentage received: {percentage}", flush=True)
    try:
        percentage_int = int(percentage)
        print(f"Using training percentage: {percentage_int}%", flush=True)
    except ValueError:
        print("Percentage must be an integer.", flush=True)
        sys.exit(1)

    # Paths
    base_dir = f"./Train_{percentage}/models/PL-fMDP/"
    analysis_file = "./analysis/count_sample_space_auto_with_safe.csv"
    safe_file = "./analysis/no_crashes.csv"
    output_file = os.path.join(base_dir, "Results", "testing_numeralia_lookup_table_frequency.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    df = pd.read_csv(analysis_file)
    print(f"Loaded {len(df)} rows from {analysis_file}", flush=True)

    feature_cols = ["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    target_col = "action"
    count_col = "count"

    missing = [c for c in feature_cols + [target_col, count_col] if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", flush=True)
        sys.exit(1)

    df = df[feature_cols + [target_col, count_col]].copy()

    for col in feature_cols:
        df[col] = df[col].astype(str).map({
            '1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
            '1.0': 1, '0.0': 0, '1.': 1, '0.': 0
        }).fillna(0).astype(int)

    # Coarser bins for performance (original)
    df['frequency_range'] = df[count_col].apply(get_frequency_range)
    freq_ranges = sorted(df['frequency_range'].unique())

    # Finer bins for safety
    df['frequency_category'] = df[count_col].apply(get_frequency_category)
    freq_categories = sorted(df['frequency_category'].unique())

    known_classes = ['change_to_left', 'change_to_right', 'cruise', 'keep']
    encoder = LabelEncoder()
    encoder.fit(known_classes)

    safe_lookup = create_safe_lookup(safe_file)
    print(f"Safe state-action pairs: {len(safe_lookup):,}", flush=True)

    # Models
    all_files = os.listdir(base_dir)
    model_tuples = [(extract_fold_number(f), f) for f in all_files if extract_fold_number(f) is not None]
    model_tuples.sort()
    model_files = [f for _, f in model_tuples]

    if not model_files:
        print("No models found!", flush=True)
        sys.exit(1)

    # Aggregators - Performance
    agg_results = {
        fr: {'precisions':[], 'recalls':[], 'f1s':[], 'accuracies':[],
             'test_times':[], 'test_sizes':[], 'time_per_row':[], 'fold_numbers':[]}
        for fr in freq_ranges
    }
    global_agg = {'precisions':[], 'recalls':[], 'f1s':[], 'accuracies':[],
                  'test_times':[], 'test_sizes':[], 'time_per_row':[]}

    # Aggregators - Safety
    disagreement_safety = {
        cat: {
            'total_disagreements': 0,
            'total_samples': 0,
            'dsafe_asafe': 0,
            'dsafe_aunsafe': 0,
            'dunsafe_asafe': 0,
            'dunsafe_aunsafe': 0,
            'by_action': {a: {'total':0, 'dsafe_asafe':0, 'dsafe_aunsafe':0,
                              'dunsafe_asafe':0, 'dunsafe_aunsafe':0}
                          for a in known_classes}
        }
        for cat in freq_categories
    }
    global_disagreement = {'total_disagreements': 0,
                           'dsafe_asafe': 0, 'dsafe_aunsafe': 0,
                           'dunsafe_asafe': 0, 'dunsafe_aunsafe': 0}

    # Header
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PL-fMDP Lookup-Table Testing Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data source: {analysis_file}\n")
        f.write(f"Safe source: {safe_file}\n")
        f.write(f"Total rows tested: {len(df)}\n")
        f.write(f"Training percentage: {percentage}%\n\n")

    # Main loop
    for fold_num, fname in model_tuples:
        pyfile = os.path.join(base_dir, fname)
        print(f"Processing Fold {fold_num} → {fname}", flush=True)

        policy = load_action_policy(pyfile)
        if policy is None:
            continue

        # Performance (original coarser bins)
        for fr in freq_ranges:
            group = df[df['frequency_range'] == fr]
            if len(group) == 0:
                continue

            X_test = group[feature_cols].copy()
            y_test_str = group[target_col].values
            y_test = encoder.transform(y_test_str)

            start_t = time.time()
            try:
                y_pred_str = policy.predict(X_test)
                y_pred = encoder.transform(y_pred_str)
            except Exception as e:
                print(f"Prediction failed for {fr}: {e}", flush=True)
                continue
            end_t = time.time()

            test_time = end_t - start_t
            test_size = len(X_test)
            time_per_row = test_time / test_size if test_size > 0 else 0

            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(y_test, y_pred)

            agg_results[fr]['precisions'].append(prec)
            agg_results[fr]['recalls'].append(rec)
            agg_results[fr]['f1s'].append(f1)
            agg_results[fr]['accuracies'].append(acc)
            agg_results[fr]['test_times'].append(test_time)
            agg_results[fr]['test_sizes'].append(test_size)
            agg_results[fr]['time_per_row'].append(time_per_row)
            agg_results[fr]['fold_numbers'].append(fold_num)

            global_agg['precisions'].append(prec)
            global_agg['recalls'].append(rec)
            global_agg['f1s'].append(f1)
            global_agg['accuracies'].append(acc)
            global_agg['test_times'].append(test_time)
            global_agg['test_sizes'].append(test_size)
            global_agg['time_per_row'].append(time_per_row)

            with open(output_file, "a") as f:
                f.write(f"\nFold {fold_num} - {fr}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Test size: {test_size:,} rows\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n")
                f.write(f"F1-score: {f1:.4f}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Test time: {test_time:.4f}s\n")
                f.write(f"Time per row: {time_per_row:.6f}s\n\n")

        # Safety analysis (finer bins)
        y_pred_str_all = policy.predict(df[feature_cols].copy())
        for i, row_idx in enumerate(df.index):
            row = df.loc[row_idx]
            true_action = row[target_col]
            pred_action = y_pred_str_all[i]

            if true_action == pred_action:
                continue

            cat = row['frequency_category']
            state_row = row[feature_cols]

            ds_safe = check_action_safety(true_action, state_row, feature_cols, safe_lookup)
            ap_safe = check_action_safety(pred_action, state_row, feature_cols, safe_lookup)

            disagreement_safety[cat]['total_disagreements'] += 1
            disagreement_safety[cat]['total_samples'] += 1
            disagreement_safety[cat]['by_action'][true_action]['total'] += 1

            global_disagreement['total_disagreements'] += 1

            if ds_safe and ap_safe:
                k = 'dsafe_asafe'
            elif ds_safe and not ap_safe:
                k = 'dsafe_aunsafe'
            elif not ds_safe and ap_safe:
                k = 'dunsafe_asafe'
            else:
                k = 'dunsafe_aunsafe'

            disagreement_safety[cat][k] += 1
            disagreement_safety[cat]['by_action'][true_action][k] += 1
            global_disagreement[k] += 1

    # Aggregated performance — your desired format
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("AGGREGATED RESULTS PER FREQUENCY RANGE (mean ± std across folds)\n")
        f.write("=" * 80 + "\n\n")

        for fr in sorted(freq_ranges):
            res = agg_results[fr]
            n = len(res['f1s'])
            if n == 0:
                f.write(f"{fr}: No valid results\n\n")
                continue

            total_samples = sum(res['test_sizes'])
            avg_size = statistics.mean(res['test_sizes']) if n > 0 else 0
            fold_list = sorted(set(res['fold_numbers']))

            mean_prec = statistics.mean(res['precisions'])
            std_prec = statistics.stdev(res['precisions']) if n > 1 else 0.0
            mean_rec = statistics.mean(res['recalls'])
            std_rec = statistics.stdev(res['recalls']) if n > 1 else 0.0
            mean_f1 = statistics.mean(res['f1s'])
            std_f1 = statistics.stdev(res['f1s']) if n > 1 else 0.0
            mean_acc = statistics.mean(res['accuracies'])
            std_acc = statistics.stdev(res['accuracies']) if n > 1 else 0.0
            mean_time = statistics.mean(res['test_times'])
            std_time = statistics.stdev(res['test_times']) if n > 1 else 0.0
            mean_tpr = statistics.mean(res['time_per_row'])

            f.write(f"{fr}:\n")
            f.write("-" * 40 + "\n")
            f.write(f" Valid folds: {n}/20 (folds: {fold_list})\n")
            f.write(f" Total samples tested: {total_samples:,}\n")
            f.write(f" Average test size per fold: {avg_size:,.1f}\n")
            f.write(f" Average test time: {mean_time:.4f}s ± {std_time:.4f}s\n")
            f.write(f" Average time per row: {mean_tpr:.6f}s\n")
            f.write(f" F1-score: {mean_f1:.4f} ± {std_f1:.4f}\n")
            f.write(f" Precision: {mean_prec:.4f} ± {std_prec:.4f}\n")
            f.write(f" Recall: {mean_rec:.4f} ± {std_rec:.4f}\n")
            f.write(f" Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write("-" * 60 + "\n\n")

        # Global
        g_n = len(global_agg['f1s'])
        if g_n > 0:
            f.write("=" * 80 + "\n")
            f.write("GLOBAL SUMMARY ACROSS ALL FREQUENCY RANGES AND FOLDS\n")
            f.write("=" * 80 + "\n\n")
            g_total = sum(global_agg['test_sizes'])
            g_avg_size = statistics.mean(global_agg['test_sizes'])
            f.write(f"Total samples tested: {g_total:,}\n")
            f.write(f"Avg test size per fold (approx): {g_avg_size:,.1f}\n")
            f.write(f"Global F1-score: {statistics.mean(global_agg['f1s']):.4f} ± {statistics.stdev(global_agg['f1s']) if g_n>1 else 0:.4f}\n")
            f.write(f"Global Precision: {statistics.mean(global_agg['precisions']):.4f} ± {statistics.stdev(global_agg['precisions']) if g_n>1 else 0:.4f}\n")
            f.write(f"Global Recall: {statistics.mean(global_agg['recalls']):.4f} ± {statistics.stdev(global_agg['recalls']) if g_n>1 else 0:.4f}\n")
            f.write(f"Global Accuracy: {statistics.mean(global_agg['accuracies']):.4f} ± {statistics.stdev(global_agg['accuracies']) if g_n>1 else 0:.4f}\n\n")

    # Safety analysis section
    with open(output_file, "a") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write("SAFETY & DISAGREEMENT ANALYSIS (finer bins)\n")
        f.write("=" * 90 + "\n\n")

        for cat in sorted(freq_categories):
            s = disagreement_safety[cat]
            td = s['total_disagreements']
            ts = s['total_samples']

            f.write(f"{cat}:\n")
            f.write(f" Total samples: {ts:,}\n")
            f.write(f" Total disagreements: {td:,} ({td/ts*100:.1f}% if ts else 0)\n")

            if td == 0:
                f.write("\n")
                continue

            net = s['dunsafe_asafe'] - s['dsafe_aunsafe']
            f.write(f" AP Safer: {s['dunsafe_asafe']:,} ({s['dunsafe_asafe']/td*100:.1f}%)\n")
            f.write(f" AP Riskier: {s['dsafe_aunsafe']:,} ({s['dsafe_aunsafe']/td*100:.1f}%)\n")
            f.write(f" Net safety improvement: {net:+d}\n")

            if cat in ["Zero (0)", "Very Low (1-4)"]:
                f.write("\nSafety Confusion Matrix:\n")
                f.write(f"Dataset Safe   AP Safe: {s['dsafe_asafe']:,}   AP Unsafe: {s['dsafe_aunsafe']:,}\n")
                f.write(f"Dataset Unsafe AP Safe: {s['dunsafe_asafe']:,}   AP Unsafe: {s['dunsafe_aunsafe']:,}\n\n")

                f.write("By dataset action:\n")
                for a in known_classes:
                    st = s['by_action'][a]
                    if st['total'] > 0:
                        f.write(f"  {a}: total {st['total']:,}   Safer {st['dunsafe_asafe']:,}   Riskier {st['dsafe_aunsafe']:,}\n")
            f.write("\n")

        # Global safety
        f.write("=" * 90 + "\n")
        f.write("GLOBAL SAFETY SUMMARY\n")
        f.write("=" * 90 + "\n\n")
        gt = global_disagreement['total_disagreements']
        if gt > 0:
            net_g = global_disagreement['dunsafe_asafe'] - global_disagreement['dsafe_aunsafe']
            f.write(f"Total disagreements: {gt:,}\n")
            f.write(f"AP Safer: {global_disagreement['dunsafe_asafe']:,} ({global_disagreement['dunsafe_asafe']/gt*100:.1f}%)\n")
            f.write(f"AP Riskier: {global_disagreement['dsafe_aunsafe']:,} ({global_disagreement['dsafe_aunsafe']/gt*100:.1f}%)\n")
            f.write(f"Net: {net_g:+d}\n")
        else:
            f.write("No disagreements found.\n")

    print(f"Results saved to {output_file}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_APs_lookup_table_frequency.py <percentage>")
        sys.exit(1)
    main(sys.argv[1])
