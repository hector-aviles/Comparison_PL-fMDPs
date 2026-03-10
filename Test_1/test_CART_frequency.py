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


# ────────────────────────────────────────────────────────────────
# LOAD MLP MODEL
# ────────────────────────────────────────────────────────────────
def load_mlp_model(model_file):
    """Load MLP model from .mlp/.pkl/.joblib"""
    try:
        model = joblib.load(model_file)
        print(f"Loaded MLP model from {model_file}", flush=True)
        return model
    except Exception as e:
        print(f"Failed to load {model_file}: {str(e)}", flush=True)
        return None


# ────────────────────────────────────────────────────────────────
# FREQUENCY BINS (same as CART version)
# ────────────────────────────────────────────────────────────────
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
    match = re.search(r'(?i)MLP[_fold]*(\d+)\.(mlp|pkl|joblib)', filename)
    if match:
        return int(match.group(1))
    return None


# ────────────────────────────────────────────────────────────────
# SAFETY FUNCTIONS (UNCHANGED)
# ────────────────────────────────────────────────────────────────
def create_safe_lookup(safe_file):
    safe_df = pd.read_csv(safe_file)

    example_cols = ['action','curr_lane','free_E','free_NE','free_NW','free_SE','free_SW','free_W']
    bool_cols = ['curr_lane','free_E','free_NE','free_NW','free_SE','free_SW','free_W']

    for col in bool_cols:
        safe_df[col] = safe_df[col].astype(str).map({
            '1':1,'0':0,'True':1,'False':0,'true':1,'false':0,
            '1.0':1,'0.0':0,'1.':1,'0.':0
        }).fillna(0).astype(int)

    return set(tuple(row) for row in safe_df[example_cols].itertuples(index=False, name=None))


def check_action_safety(action, state_row, feature_cols, safe_lookup):
    state_values = [state_row[col] for col in feature_cols]
    return tuple([action] + state_values) in safe_lookup


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main(percentage):

    print(f"Percentage received: {percentage}", flush=True)

    try:
        percentage_int = int(percentage)
    except ValueError:
        print("Percentage must be integer.")
        sys.exit(1)

    mlp_path = f"./Train_{percentage}/models/MLP/"
    analysis_file = "./analysis/count_sample_space_auto_with_safe.csv"
    safe_file = "./analysis/no_crashes.csv"
    numeralia = os.path.join(mlp_path, "Results", "testing_mlp_frequency.txt")

    os.makedirs(os.path.dirname(numeralia), exist_ok=True)

    full_data = pd.read_csv(analysis_file)

    feature_columns = ["curr_lane","free_E","free_NE","free_NW","free_SE","free_SW","free_W"]
    target_col = "action"
    count_col = "count"

    full_data = full_data[feature_columns + [target_col, count_col]].copy()

    for col in feature_columns:
        full_data[col] = full_data[col].astype(str).map({
            '1':1,'0':0,'True':1,'False':0,'true':1,'false':0,
            '1.0':1,'0.0':0,'1.':1,'0.':0
        }).fillna(0).astype(int)

    full_data['frequency_range'] = full_data[count_col].apply(get_frequency_range)
    full_data['frequency_category'] = full_data[count_col].apply(get_frequency_category)

    frequency_ranges = sorted(full_data['frequency_range'].unique())
    freq_categories = sorted(full_data['frequency_category'].unique())

    safe_lookup = create_safe_lookup(safe_file)

    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left','change_to_right','cruise','keep'])

    # ── Discover models ─────────────────────────
    model_files = []
    for f in os.listdir(mlp_path):
        if f.lower().endswith(('.mlp','.pkl','.joblib')):
            fold = extract_fold_number(f)
            if fold is not None:
                model_files.append((fold,f))

    model_files.sort()

    print(f"Found {len(model_files)} MLP models", flush=True)

    # ── Aggregators ─────────────────────────
    results = {fr:{'precisions':[],'recalls':[],'f1_scores':[],'accuracies':[],
                   'test_times':[],'test_sizes':[],'fold_numbers':[]} for fr in frequency_ranges}

    disagreement_safety = {cat:{'total_disagreements':0,'dsafe_asafe':0,'dsafe_aunsafe':0,
                               'dunsafe_asafe':0,'dunsafe_aunsafe':0}
                           for cat in freq_categories}


    # ────────────────────────────────────────
    # TEST LOOP
    # ────────────────────────────────────────
    for fold_num,model_filename in model_files:

        print(f"\nProcessing Fold {fold_num}", flush=True)

        model = load_mlp_model(os.path.join(mlp_path,model_filename))
        if model is None:
            continue

        for fr in frequency_ranges:

            subset = full_data[full_data['frequency_range']==fr]
            if len(subset)==0:
                continue

            X_test = subset[feature_columns].values.astype(float)
            y_test = encoder.transform(subset[target_col])

            start = time.time()
            y_pred = model.predict(X_test)
            end = time.time()

            y_pred = y_pred.astype(int)

            precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
            recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
            f1 = f1_score(y_test,y_pred,average='weighted',zero_division=0)
            acc = accuracy_score(y_test,y_pred)

            results[fr]['precisions'].append(precision)
            results[fr]['recalls'].append(recall)
            results[fr]['f1_scores'].append(f1)
            results[fr]['accuracies'].append(acc)
            results[fr]['test_times'].append(end-start)
            results[fr]['test_sizes'].append(len(subset))
            results[fr]['fold_numbers'].append(fold_num)

        # ── Safety disagreements (global per fold)
        y_pred_all = model.predict(full_data[feature_columns].values.astype(float))

        for i,idx in enumerate(full_data.index):

            true_action = full_data.loc[idx,target_col]
            pred_action = encoder.inverse_transform([int(y_pred_all[i])])[0]

            if true_action == pred_action:
                continue

            cat = full_data.loc[idx,'frequency_category']
            state_row = full_data.loc[idx,feature_columns]

            ds_safe = check_action_safety(true_action,state_row,feature_columns,safe_lookup)
            mlp_safe = check_action_safety(pred_action,state_row,feature_columns,safe_lookup)

            disagreement_safety[cat]['total_disagreements'] += 1

            if ds_safe and mlp_safe:
                disagreement_safety[cat]['dsafe_asafe']+=1
            elif ds_safe and not mlp_safe:
                disagreement_safety[cat]['dsafe_aunsafe']+=1
            elif not ds_safe and mlp_safe:
                disagreement_safety[cat]['dunsafe_asafe']+=1
            else:
                disagreement_safety[cat]['dunsafe_aunsafe']+=1


    print("\nDone.")
