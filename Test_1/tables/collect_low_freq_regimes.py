import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_metrics_file(filepath, model_name):
    """
    Parse a model's testing frequency file to extract:
    1. Performance metrics (Precision, Recall, F1) with std dev
    2. Safety analysis for combined Low-Frequency (0-4) category
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    result = {
        'model': model_name,
        'precision': None, 'precision_std': None,
        'recall': None, 'recall_std': None,
        'f1': None, 'f1_std': None,
        # Combined low-frequency metrics
        'low_total': 0, 'low_disagreements': 0,
        'low_safer': 0, 'low_riskier': 0, 'low_net': 0,
        'global_disagreements': 0,
        'global_safer': 0, 'global_riskier': 0, 'global_net': 0
    }
    
    # Extract performance metrics from AGGREGATED RESULTS section
    perf_patterns = {
        'precision': r'Precision:\s*([0-9.]+)\s*±\s*([0-9.]+)',
        'recall': r'Recall:\s*([0-9.]+)\s*±\s*([0-9.]+)',
        'f1': r'F1-score:\s*([0-9.]+)\s*±\s*([0-9.]+)'
    }
    
    for metric, pattern in perf_patterns.items():
        match = re.search(pattern, content)
        if match:
            result[metric] = float(match.group(1))
            result[f'{metric}_std'] = float(match.group(2))
    
    # If not found in aggregated, try global summary
    if result['f1'] is None:
        global_patterns = {
            'precision': r'Global Precision:\s*([0-9.]+)\s*±\s*([0-9.]+)',
            'recall': r'Global Recall:\s*([0-9.]+)\s*±\s*([0-9.]+)',
            'f1': r'Global F1-score:\s*([0-9.]+)\s*±\s*([0-9.]+)'
        }
        for metric, pattern in global_patterns.items():
            match = re.search(pattern, content)
            if match:
                result[metric] = float(match.group(1))
                result[f'{metric}_std'] = float(match.group(2))
    
    # Try to find combined Low-Frequency (0-4) section first
    # Some files might already have a combined "Low (0-4)" section
    low_combined_pattern = r'Low \(0-4\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                          r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Safer:\s*([0-9,]+).*?' + \
                          r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Riskier:\s*([0-9,]+).*?' + \
                          r'Net safety improvement:\s*([+-]?\d+)'
    
    low_match = re.search(low_combined_pattern, content, re.DOTALL)
    if low_match:
        result['low_total'] = int(low_match.group(1).replace(',', ''))
        result['low_disagreements'] = int(low_match.group(2).replace(',', ''))
        result['low_safer'] = int(low_match.group(3).replace(',', ''))
        result['low_riskier'] = int(low_match.group(4).replace(',', ''))
        result['low_net'] = int(low_match.group(5))
    else:
        # If not combined, try to extract Zero and Very Low separately and combine them
        # Extract Zero (0) category
        zero_pattern = r'Zero \(0\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                      r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Safer:\s*([0-9,]+).*?' + \
                      r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Riskier:\s*([0-9,]+).*?' + \
                      r'Net safety improvement:\s*([+-]?\d+)'
        
        zero_match = re.search(zero_pattern, content, re.DOTALL)
        
        # Extract Very Low (1-4) category
        verylow_pattern = r'Very Low \(1-4\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                         r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Safer:\s*([0-9,]+).*?' + \
                         r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Riskier:\s*([0-9,]+).*?' + \
                         r'Net safety improvement:\s*([+-]?\d+)'
        
        verylow_match = re.search(verylow_pattern, content, re.DOTALL)
        
        if zero_match and verylow_match:
            # Combine Zero and Very Low
            zero_total = int(zero_match.group(1).replace(',', ''))
            zero_disagreements = int(zero_match.group(2).replace(',', ''))
            zero_safer = int(zero_match.group(3).replace(',', ''))
            zero_riskier = int(zero_match.group(4).replace(',', ''))
            zero_net = int(zero_match.group(5))
            
            verylow_total = int(verylow_match.group(1).replace(',', ''))
            verylow_disagreements = int(verylow_match.group(2).replace(',', ''))
            verylow_safer = int(verylow_match.group(3).replace(',', ''))
            verylow_riskier = int(verylow_match.group(4).replace(',', ''))
            verylow_net = int(verylow_match.group(5))
            
            result['low_total'] = zero_total + verylow_total
            result['low_disagreements'] = zero_disagreements + verylow_disagreements
            result['low_safer'] = zero_safer + verylow_safer
            result['low_riskier'] = zero_riskier + verylow_riskier
            result['low_net'] = zero_net + verylow_net
    
    # Extract global safety summary
    global_pattern = r'GLOBAL SAFETY SUMMARY.*?Total disagreements:\s*([0-9,]+).*?' + \
                     r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Safer:\s*([0-9,]+).*?' + \
                     r'(?:AP|CART|NB|LR|RF|XGBoost|MLP)\s+Riskier:\s*([0-9,]+).*?' + \
                     r'Net:\s*([+-]?\d+)'
    
    global_match = re.search(global_pattern, content, re.DOTALL)
    if global_match:
        result['global_disagreements'] = int(global_match.group(1).replace(',', ''))
        result['global_safer'] = int(global_match.group(2).replace(',', ''))
        result['global_riskier'] = int(global_match.group(3).replace(',', ''))
        result['global_net'] = int(global_match.group(4))
    
    return result

def format_metric(value, std):
    """Format metric with ± std"""
    if value is None or std is None:
        return "N/A"
    return f"{value:.4f}±{std:.4f}"

def format_safety(safer, riskier, total, net):
    """Format safety metrics"""
    if total == 0:
        safer_pct = riskier_pct = 0
    else:
        safer_pct = (safer / total * 100) if total > 0 else 0
        riskier_pct = (riskier / total * 100) if total > 0 else 0
    
    return {
        'safer': f"{safer} ({safer_pct:.1f}%)",
        'riskier': f"{riskier} ({riskier_pct:.1f}%)",
        'net': f"{net:+d}"
    }

def main():
    percentages = ['01', '50', '100']
    models = [
        ('APs', 'PL-fMDP', 'testing_numeralia_lookup_table_frequency.txt'),
        ('LR', 'LR', 'testing_lr_frequency.txt'),
        ('CART', 'CART', 'testing_cart_frequency.txt'),
        ('NB', 'NB', 'testing_nb_frequency.txt'),
        ('RF', 'RF', 'testing_rf_frequency.txt'),
        ('XGBoost', 'XGBoost', 'testing_xgboost_frequency.txt'),
        ('MLP', 'MLP', 'testing_mlp_frequency.txt')
    ]
    
    # Collect all results
    all_results = {}
    
    for pct in percentages:
        pct_results = {}
        for display_name, model_folder, filename in models:
            filepath = f"../Train_{pct}/models/{model_folder}/Results/{filename}"
            result = parse_metrics_file(filepath, display_name)
            if result:
                pct_results[display_name] = result
        all_results[pct] = pct_results
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append("\\begin{table}[!htb]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Average and standard deviation for precision-recall and F1 scores with safety analysis for low-frequency regimes (count 0-4). Numbers are rounded to 4 decimal places.}")
    latex_output.append("\\label{tab:Precision_recall_safety}")
    latex_output.append("\\small")
    latex_output.append("\\begin{tabular}{lccccc}")
    latex_output.append("\\toprule")
    latex_output.append("\\multicolumn{5}{c}{Safety Analysis Across Training Percentages} \\\\")
    latex_output.append("\\midrule")
    
    for pct in percentages:
        latex_output.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{Training percentage {pct}\\%}}}} \\\\")
        latex_output.append("\\midrule")
        latex_output.append("Model & Precision & Recall & F1 & \\multicolumn{2}{c}{Low-Frequency (0-4)} \\\\")
        latex_output.append(" & & & & Safer/Riskier & Net \\\\")
        latex_output.append("\\midrule")
        
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                
                # Performance metrics
                prec_str = format_metric(r['precision'], r['precision_std'])
                rec_str = format_metric(r['recall'], r['recall_std'])
                f1_str = format_metric(r['f1'], r['f1_std'])
                
                # Combined low-frequency safety
                low_safety = format_safety(
                    r['low_safer'], r['low_riskier'], 
                    r['low_disagreements'], r['low_net']
                )
                
                row = f"{display_name} & {prec_str} & {rec_str} & {f1_str} & "
                row += f"{low_safety['safer']}/{low_safety['riskier']} & {low_safety['net']} \\\\"
                latex_output.append(row)
        
        latex_output.append("\\midrule")
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Write LaTeX table to file
    with open("safety_analysis_table_combined.tex", "w") as f:
        f.write("\n".join(latex_output))
    
    # Also generate a CSV summary
    csv_rows = []
    for pct in percentages:
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                csv_rows.append({
                    'percentage': pct,
                    'model': display_name,
                    'precision': r['precision'],
                    'precision_std': r['precision_std'],
                    'recall': r['recall'],
                    'recall_std': r['recall_std'],
                    'f1': r['f1'],
                    'f1_std': r['f1_std'],
                    'low_safer': r['low_safer'],
                    'low_riskier': r['low_riskier'],
                    'low_net': r['low_net'],
                    'global_safer': r['global_safer'],
                    'global_riskier': r['global_riskier'],
                    'global_net': r['global_net']
                })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv("safety_analysis_summary_combined.csv", index=False)
    
    # Print summary to console
    print("\n" + "="*100)
    print("SAFETY ANALYSIS SUMMARY - COMBINED LOW-FREQUENCY (0-4)")
    print("="*100)
    
    for pct in percentages:
        print(f"\nTraining {pct}%:")
        print("-"*80)
        print(f"{'Model':<10} {'Low Net':>10} {'Low Safer':>12} {'Low Riskier':>14} | {'Global Net':>12} {'Global Safer':>14} {'Global Riskier':>14}")
        print("-"*80)
        
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                print(f"{display_name:<10} {r['low_net']:>10d} {r['low_safer']:>12,d} {r['low_riskier']:>14,d} | "
                      f"{r['global_net']:>12d} {r['global_safer']:>14,d} {r['global_riskier']:>14,d}")
    
    print("\n" + "="*100)
    print("LaTeX table saved to: safety_analysis_table_combined.tex")
    print("CSV summary saved to: safety_analysis_summary_combined.csv")
    print("="*100)

if __name__ == "__main__":
    main()
