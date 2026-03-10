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
    """Dynamically load ActionPolicy class from .py file"""
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

def get_frequency_category(count):
    """Categorize frequency count into detailed ranges - SEPARATING 0 AND 1-4"""
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
    """Extract fold number from filename like APClassifier_5_lookup_table.py"""
    match = re.search(r'APClassifier_(\d+)_lookup_table\.py', filename)
    if match:
        return int(match.group(1))
    return None

def create_safe_lookup(safe_file):
    """Create a lookup set for safe state-action pairs from no_crashes.csv"""
    safe_df = pd.read_csv(safe_file)
    example_cols = ['action', 'curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    
    # Normalize boolean columns to 0/1 for consistent comparison
    bool_cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
    for col in bool_cols:
        safe_df[col] = safe_df[col].astype(str).map({
            '1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
            '1.0': 1, '0.0': 0, '1.': 1, '0.': 0
        }).fillna(0).astype(int)
    
    return set(tuple(row) for row in safe_df[example_cols].itertuples(index=False, name=None))

def check_action_safety(action, state_row, feature_cols, safe_lookup):
    """Check if a given action in a given state is safe"""
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
    disagreements_file = os.path.join(base_dir, "Results", "disagreements_safety_analysis.csv")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    if not os.path.exists(analysis_file):
        print(f"Analysis file not found: {analysis_file}", flush=True)
        sys.exit(1)
    
    if not os.path.exists(safe_file):
        print(f"Safe file not found: {safe_file}", flush=True)
        sys.exit(1)

    df = pd.read_csv(analysis_file)
    print(f"Loaded {len(df)} rows from {analysis_file}", flush=True)
    
    # Create safe lookup
    safe_lookup = create_safe_lookup(safe_file)
    print(f"Created lookup set with {len(safe_lookup):,} safe state-action pairs from {safe_file}", flush=True)

    # Required columns
    feature_cols = ["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    target_col = "action"
    count_col = "count"

    missing = [c for c in feature_cols + [target_col, count_col] if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}", flush=True)
        sys.exit(1)

    # Keep needed columns
    cols_to_keep = feature_cols + [target_col, count_col]
    df = df[cols_to_keep].copy()

    # Normalize features to 0/1 integers
    for col in feature_cols:
        df[col] = df[col].astype(str).map({
            '1': 1, '0': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
            '1.0': 1, '0.0': 0, '1.': 1, '0.': 0
        }).fillna(0).astype(int)

    # Add detailed frequency category (NOW WITH SEPARATE 0 AND 1-4)
    df['frequency_category'] = df[count_col].apply(get_frequency_category)

    # Known classes
    known_classes = ['change_to_left', 'change_to_right', 'cruise', 'keep']
    encoder = LabelEncoder()
    encoder.fit(known_classes)

    # Get sorted frequency categories (will include "Zero (0)" and "Very Low (1-4)" separately)
    freq_categories = sorted(df['frequency_category'].unique())
    print(f"Frequency categories found: {freq_categories}", flush=True)
    
    # Print counts for zero and very low categories specifically
    zero_count = len(df[df[count_col] == 0])
    very_low_count = len(df[(df[count_col] >= 1) & (df[count_col] <= 4)])
    print(f"Zero-count examples (count=0): {zero_count:,}", flush=True)
    print(f"Very low-count examples (1-4): {very_low_count:,}", flush=True)

    # Find and sort model files by fold number
    all_files = os.listdir(base_dir)
    model_tuples = []
    for fname in all_files:
        fold_num = extract_fold_number(fname)
        if fold_num is not None:
            model_tuples.append((fold_num, fname))

    # Sort by fold number
    model_tuples.sort()
    model_files = [fname for _, fname in model_tuples]

    print(f"Found {len(model_files)} matching lookup-table models "
          f"(folds: {[n for n,_ in model_tuples]})", flush=True)

    if len(model_files) == 0:
        print("No matching lookup table models found!", flush=True)
        sys.exit(1)

    # Aggregated results per frequency category
    agg_results = {
        cat: {
            'precisions': [], 'recalls': [], 'f1s': [], 'accuracies': [],
            'test_times': [], 'test_sizes': [], 'time_per_row': [],
            'fold_numbers': []
        }
        for cat in freq_categories
    }

    # DISAGREEMENT SAFETY ANALYSIS - 2x2 matrix for each frequency category
    disagreement_safety = {
        cat: {
            'total_disagreements': 0,
            'total_samples': 0,
            'dataset_safe_ap_safe': 0,
            'dataset_safe_ap_unsafe': 0,
            'dataset_unsafe_ap_safe': 0,
            'dataset_unsafe_ap_unsafe': 0,
            'by_action': {
                action: {
                    'total': 0,
                    'dataset_safe_ap_safe': 0,
                    'dataset_safe_ap_unsafe': 0,
                    'dataset_unsafe_ap_safe': 0,
                    'dataset_unsafe_ap_unsafe': 0
                } for action in known_classes
            },
            'detailed_examples': []
        }
        for cat in freq_categories
    }

    # Global aggregator
    global_agg = {
        'precisions': [], 'recalls': [], 'f1s': [], 'accuracies': [],
        'test_times': [], 'test_sizes': [], 'time_per_row': []
    }
    
    # Global disagreement safety
    global_disagreement = {
        'total_disagreements': 0,
        'dataset_safe_ap_safe': 0,
        'dataset_safe_ap_unsafe': 0,
        'dataset_unsafe_ap_safe': 0,
        'dataset_unsafe_ap_unsafe': 0
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PL-fMDP Lookup-Table Testing Results by Detailed Frequency Category\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data source: {analysis_file}\n")
        f.write(f"Safe examples source: {safe_file}\n")
        f.write(f"Total rows tested: {len(df)}\n")
        f.write(f"Training percentage: {percentage}%\n\n")
        f.write("NOTE: 'Zero (0)' and 'Very Low (1-4)' are now SEPARATED for detailed analysis\n")
        f.write("      Focus on disagreements (AP prediction ≠ dataset action)\n\n")

        f.write("Frequency Category Distribution:\n")
        for cat in freq_categories:
            cnt = len(df[df['frequency_category'] == cat])
            f.write(f"  {cat}: {cnt:,} samples\n")
        f.write("\n" + "=" * 80 + "\n\n")

    # Process each model
    for fold_num, fname in model_tuples:
        pyfile = os.path.join(base_dir, fname)
        print(f"\n{'='*70}", flush=True)
        print(f"Processing Fold {fold_num} → {fname}", flush=True)

        policy = load_action_policy(pyfile)
        if policy is None:
            continue

        fold_has_data = False

        for cat in freq_categories:
            group = df[df['frequency_category'] == cat]
            if len(group) == 0:
                print(f"  {cat}: no samples → skipping", flush=True)
                continue

            X_test = group[feature_cols].copy()
            y_test_str = group[target_col].values

            try:
                y_test = encoder.transform(y_test_str)
            except ValueError as e:
                print(f"  {cat}: Label encoding error → {e}", flush=True)
                continue

            test_size = len(X_test)
            disagreement_safety[cat]['total_samples'] += test_size

            start_t = time.time()
            try:
                y_pred_str = policy.predict(X_test)
                y_pred = encoder.transform(y_pred_str)
            except Exception as e:
                print(f"  Prediction failed for {cat} in fold {fold_num}: {str(e)}", flush=True)
                continue

            end_t = time.time()
            test_time = end_t - start_t
            time_per_row = test_time / test_size if test_size > 0 else 0.0

            # Calculate standard metrics
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            acc = accuracy_score(y_test, y_pred)

            # Store per-category
            agg_results[cat]['precisions'].append(prec)
            agg_results[cat]['recalls'].append(rec)
            agg_results[cat]['f1s'].append(f1)
            agg_results[cat]['accuracies'].append(acc)
            agg_results[cat]['test_times'].append(test_time)
            agg_results[cat]['test_sizes'].append(test_size)
            agg_results[cat]['time_per_row'].append(time_per_row)
            agg_results[cat]['fold_numbers'].append(fold_num)

            # Global
            global_agg['precisions'].append(prec)
            global_agg['recalls'].append(rec)
            global_agg['f1s'].append(f1)
            global_agg['accuracies'].append(acc)
            global_agg['test_times'].append(test_time)
            global_agg['test_sizes'].append(test_size)
            global_agg['time_per_row'].append(time_per_row)

            # DISAGREEMENT SAFETY ANALYSIS
            fold_disagreements = 0
            for i, (true_action, pred_action, idx) in enumerate(zip(y_test_str, y_pred_str, group.index)):
                if true_action != pred_action:  # This is a disagreement
                    fold_disagreements += 1
                    
                    # Get the state row
                    state_row = group.loc[idx, feature_cols]
                    
                    # Check safety
                    dataset_safe = check_action_safety(true_action, state_row, feature_cols, safe_lookup)
                    ap_safe = check_action_safety(pred_action, state_row, feature_cols, safe_lookup)
                    
                    # Update statistics
                    disagreement_safety[cat]['total_disagreements'] += 1
                    disagreement_safety[cat]['by_action'][true_action]['total'] += 1
                    
                    global_disagreement['total_disagreements'] += 1
                    
                    # Categorize
                    if dataset_safe and ap_safe:
                        disagreement_safety[cat]['dataset_safe_ap_safe'] += 1
                        disagreement_safety[cat]['by_action'][true_action]['dataset_safe_ap_safe'] += 1
                        global_disagreement['dataset_safe_ap_safe'] += 1
                        category = "Dataset Safe + AP Safe"
                    elif dataset_safe and not ap_safe:
                        disagreement_safety[cat]['dataset_safe_ap_unsafe'] += 1
                        disagreement_safety[cat]['by_action'][true_action]['dataset_safe_ap_unsafe'] += 1
                        global_disagreement['dataset_safe_ap_unsafe'] += 1
                        category = "Dataset Safe + AP Unsafe"
                    elif not dataset_safe and ap_safe:
                        disagreement_safety[cat]['dataset_unsafe_ap_safe'] += 1
                        disagreement_safety[cat]['by_action'][true_action]['dataset_unsafe_ap_safe'] += 1
                        global_disagreement['dataset_unsafe_ap_safe'] += 1
                        category = "Dataset Unsafe + AP Safe"
                    else:
                        disagreement_safety[cat]['dataset_unsafe_ap_unsafe'] += 1
                        disagreement_safety[cat]['by_action'][true_action]['dataset_unsafe_ap_unsafe'] += 1
                        global_disagreement['dataset_unsafe_ap_unsafe'] += 1
                        category = "Dataset Unsafe + AP Unsafe"
                    
                    # Store examples
                    if len(disagreement_safety[cat]['detailed_examples']) < 10:
                        disagreement_safety[cat]['detailed_examples'].append({
                            'fold': fold_num,
                            'frequency_category': cat,
                            'count': group.loc[idx, count_col],
                            'true_action': true_action,
                            'pred_action': pred_action,
                            'dataset_safe': dataset_safe,
                            'ap_safe': ap_safe,
                            'category': category,
                            'state': {col: state_row[col] for col in feature_cols}
                        })

            fold_has_data = True

            safe_pct_ap = (disagreement_safety[cat]['dataset_unsafe_ap_safe'] / max(1, disagreement_safety[cat]['total_disagreements']) * 100)
            print(f"  {cat}: {test_size:,} samples | F1={f1:.4f} | Disagreements={fold_disagreements} | AP Safer={safe_pct_ap:.1f}%", flush=True)

            # Write per-fold detail
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\nFold {fold_num} - {cat}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Test size: {test_size:,} rows\n")
                f.write(f"Disagreements: {fold_disagreements} ({(fold_disagreements/test_size*100):.1f}%)\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n")
                f.write(f"F1-score: {f1:.4f}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Test time: {test_time:.4f}s\n")
                f.write(f"Time per row: {time_per_row:.6f}s\n\n")

        if fold_has_data:
            print(f"  Fold {fold_num} completed with data", flush=True)

    # Write detailed disagreement safety analysis
    with open(disagreements_file, "w", encoding="utf-8") as f:
        f.write("DETAILED DISAGREEMENT SAFETY ANALYSIS - WITH ZERO AND VERY LOW SEPARATED\n")
        f.write("=" * 90 + "\n\n")
        
        # First, highlight the zero and very low categories
        f.write("🔍 FOCUS ON LOW-FREQUENCY REGIMES:\n")
        f.write("-" * 40 + "\n")
        for cat in ["Zero (0)", "Very Low (1-4)"]:
            if cat in disagreement_safety:
                stats = disagreement_safety[cat]
                f.write(f"\n{cat}:\n")
                f.write(f"  Total samples: {stats['total_samples']:,}\n")
                f.write(f"  Total disagreements: {stats['total_disagreements']:,}\n")
                if stats['total_disagreements'] > 0:
                    f.write(f"  AP Safer (Dataset Unsafe + AP Safe): {stats['dataset_unsafe_ap_safe']} ({stats['dataset_unsafe_ap_safe']/stats['total_disagreements']*100:.1f}%)\n")
                    f.write(f"  AP Riskier (Dataset Safe + AP Unsafe): {stats['dataset_safe_ap_unsafe']} ({stats['dataset_safe_ap_unsafe']/stats['total_disagreements']*100:.1f}%)\n")
        f.write("\n" + "=" * 90 + "\n\n")
        
        # Full analysis for all categories
        for cat in freq_categories:
            stats = disagreement_safety[cat]
            if stats['total_disagreements'] == 0:
                f.write(f"\n{'='*60}\n")
                f.write(f"{cat}: No disagreements\n")
                continue
                
            f.write(f"\n{'='*60}\n")
            f.write(f"FREQUENCY CATEGORY: {cat}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Total samples: {stats['total_samples']:,}\n")
            f.write(f"Total disagreements: {stats['total_disagreements']:,} "
                   f"({stats['total_disagreements']/stats['total_samples']*100:.1f}% of samples)\n\n")
            
            f.write("SAFETY CONFUSION MATRIX (Dataset Action vs AP Prediction):\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'':20} {'AP Safe':>15} {'AP Unsafe':>15} {'Total':>10}\n")
            f.write("-" * 70 + "\n")
            
            ds_safe_total = stats['dataset_safe_ap_safe'] + stats['dataset_safe_ap_unsafe']
            ds_unsafe_total = stats['dataset_unsafe_ap_safe'] + stats['dataset_unsafe_ap_unsafe']
            
            f.write(f"{'Dataset Safe':20} {stats['dataset_safe_ap_safe']:15,d} {stats['dataset_safe_ap_unsafe']:15,d} {ds_safe_total:10,d}\n")
            f.write(f"{'Dataset Unsafe':20} {stats['dataset_unsafe_ap_safe']:15,d} {stats['dataset_unsafe_ap_unsafe']:15,d} {ds_unsafe_total:10,d}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Total':20} {stats['dataset_safe_ap_safe'] + stats['dataset_unsafe_ap_safe']:15,d} "
                   f"{stats['dataset_safe_ap_unsafe'] + stats['dataset_unsafe_ap_unsafe']:15,d} {stats['total_disagreements']:10,d}\n\n")
            
            # Percentages
            f.write("PERCENTAGES (of total disagreements):\n")
            f.write("-" * 50 + "\n")
            total = stats['total_disagreements']
            f.write(f"✅ Dataset Unsafe + AP Safe: {stats['dataset_unsafe_ap_safe']/total*100:6.2f}% - AP improves safety\n")
            f.write(f"⚠️ Dataset Safe + AP Unsafe:   {stats['dataset_safe_ap_unsafe']/total*100:6.2f}% - AP reduces safety\n")
            f.write(f"👍 Dataset Safe + AP Safe:     {stats['dataset_safe_ap_safe']/total*100:6.2f}% - Both safe\n")
            f.write(f"👎 Dataset Unsafe + AP Unsafe: {stats['dataset_unsafe_ap_unsafe']/total*100:6.2f}% - Both unsafe\n\n")
            
            # Key insight for research question
            f.write("🔬 KEY INSIGHT FOR MODEL-BASED RL:\n")
            improvement = stats['dataset_unsafe_ap_safe'] - stats['dataset_safe_ap_unsafe']
            f.write(f"  Net safety improvement (AP Safe in Unsafe - AP Unsafe in Safe): {improvement:+d}\n")
            if improvement > 0:
                f.write(f"  → APs make SAFER choices overall in {cat} regime ✅\n")
            elif improvement < 0:
                f.write(f"  → APs make RISKIER choices overall in {cat} regime ⚠️\n")
            else:
                f.write(f"  → APs have neutral safety impact in {cat} regime\n")
            f.write("\n")
            
            # Breakdown by action
            f.write("BREAKDOWN BY DATASET ACTION:\n")
            f.write("-" * 60 + "\n")
            for action in known_classes:
                action_stats = stats['by_action'][action]
                if action_stats['total'] > 0:
                    f.write(f"\n{action}:\n")
                    f.write(f"  Total disagreements: {action_stats['total']}\n")
                    f.write(f"  AP Safer (Dataset Unsafe + AP Safe): {action_stats['dataset_unsafe_ap_safe']} "
                           f"({action_stats['dataset_unsafe_ap_safe']/action_stats['total']*100:5.1f}%)\n")
                    f.write(f"  AP Riskier (Dataset Safe + AP Unsafe): {action_stats['dataset_safe_ap_unsafe']} "
                           f"({action_stats['dataset_safe_ap_unsafe']/action_stats['total']*100:5.1f}%)\n")
            
            # Examples
            if stats['detailed_examples']:
                f.write("\n\nEXAMPLE DISAGREEMENTS:\n")
                f.write("-" * 60 + "\n")
                for i, ex in enumerate(stats['detailed_examples'], 1):
                    f.write(f"\n{i}. Fold {ex['fold']} (count={ex['count']}):\n")
                    f.write(f"   True: {ex['true_action']} ({'Safe' if ex['dataset_safe'] else 'Unsafe'}) → "
                           f"Pred: {ex['pred_action']} ({'Safe' if ex['ap_safe'] else 'Unsafe'})\n")
        
        # Global summary
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("GLOBAL SUMMARY ACROSS ALL CATEGORIES\n")
        f.write("=" * 90 + "\n\n")
        
        g = global_disagreement
        if g['total_disagreements'] > 0:
            f.write(f"Total disagreements across all folds: {g['total_disagreements']:,}\n\n")
            f.write("Global Safety Impact:\n")
            f.write(f"  ✅ AP Safer (Dataset Unsafe + AP Safe): {g['dataset_unsafe_ap_safe']} ({g['dataset_unsafe_ap_safe']/g['total_disagreements']*100:.1f}%)\n")
            f.write(f"  ⚠️ AP Riskier (Dataset Safe + AP Unsafe): {g['dataset_safe_ap_unsafe']} ({g['dataset_safe_ap_unsafe']/g['total_disagreements']*100:.1f}%)\n")
            f.write(f"  Net safety improvement: {g['dataset_unsafe_ap_safe'] - g['dataset_safe_ap_unsafe']:+d}\n")

    # Write aggregated results to main output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("AGGREGATED RESULTS PER FREQUENCY CATEGORY\n")
        f.write("=" * 80 + "\n\n")

        for cat in freq_categories:
            res = agg_results[cat]
            n = len(res['f1s'])
            if n == 0:
                f.write(f"{cat}: No valid results\n")
                continue

            mean_f1 = statistics.mean(res['f1s'])
            std_f1 = statistics.stdev(res['f1s']) if n > 1 else 0.0
            mean_prec = statistics.mean(res['precisions'])
            std_prec = statistics.stdev(res['precisions']) if n > 1 else 0.0
            mean_rec = statistics.mean(res['recalls'])
            std_rec = statistics.stdev(res['recalls']) if n > 1 else 0.0
            mean_acc = statistics.mean(res['accuracies'])
            std_acc = statistics.stdev(res['accuracies']) if n > 1 else 0.0
            total_samples = sum(res['test_sizes'])

            f.write(f"{cat}:\n")
            f.write(f"  Valid folds: {n}/20\n")
            f.write(f"  Total samples: {total_samples:,}\n")
            f.write(f"  Total disagreements: {disagreement_safety[cat]['total_disagreements']:,} "
                   f"({disagreement_safety[cat]['total_disagreements']/total_samples*100:.1f}%)\n\n")
            
            f.write(f"  SAFETY IMPACT METRICS:\n")
            d = disagreement_safety[cat]
            if d['total_disagreements'] > 0:
                ap_safer_pct = d['dataset_unsafe_ap_safe']/d['total_disagreements']*100
                ap_riskier_pct = d['dataset_safe_ap_unsafe']/d['total_disagreements']*100
                f.write(f"    ✅ AP Safer: {d['dataset_unsafe_ap_safe']:5,d} ({ap_safer_pct:5.1f}%)\n")
                f.write(f"    ⚠️ AP Riskier: {d['dataset_safe_ap_unsafe']:5,d} ({ap_riskier_pct:5.1f}%)\n")
                f.write(f"    Net safety: {d['dataset_unsafe_ap_safe'] - d['dataset_safe_ap_unsafe']:+d}\n\n")
            
            f.write(f"  Performance metrics:\n")
            f.write(f"    F1-score: {mean_f1:.4f} ± {std_f1:.4f}\n")
            f.write(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write("-" * 60 + "\n\n")

    print(f"\nTesting completed. Results saved to:\n  {output_file}\n", flush=True)
    print(f"Detailed disagreement safety analysis saved to:\n  {disagreements_file}\n", flush=True)
    
    # Print focused summary for research question
    print("\n" + "="*80, flush=True)
    print("🔬 ANSWERING YOUR RESEARCH QUESTION: Model-based RL in Low-Frequency Regimes", flush=True)
    print("="*80, flush=True)
    
    for cat in ["Zero (0)", "Very Low (1-4)"]:
        if cat in disagreement_safety:
            d = disagreement_safety[cat]
            if d['total_disagreements'] > 0:
                print(f"\n{cat}:", flush=True)
                print(f"  Total disagreements: {d['total_disagreements']:,}", flush=True)
                print(f"  ✅ AP makes SAFER choices: {d['dataset_unsafe_ap_safe']} ({d['dataset_unsafe_ap_safe']/d['total_disagreements']*100:.1f}%)", flush=True)
                print(f"  ⚠️ AP makes RISKIER choices: {d['dataset_safe_ap_unsafe']} ({d['dataset_safe_ap_unsafe']/d['total_disagreements']*100:.1f}%)", flush=True)
                
                net = d['dataset_unsafe_ap_safe'] - d['dataset_safe_ap_unsafe']
                if net > 0:
                    print(f"  ✨ CONCLUSION: APs improve safety in {cat} regime (net +{net})", flush=True)
                elif net < 0:
                    print(f"  ⚠️ CONCLUSION: APs reduce safety in {cat} regime (net {net})", flush=True)
                else:
                    print(f"  ➖ CONCLUSION: APs have neutral safety impact in {cat} regime", flush=True)
    
    print("\n" + "="*80, flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_APs_lookup_table.py <percentage>")
        print("Example: python3 test_APs_lookup_table.py 100")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
