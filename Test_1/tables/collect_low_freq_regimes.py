import sys
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

    # Map display names to the acronyms used in the files
    acronym_map = {
        'APs': 'AP',
        'LR': 'LR',
        'CART': 'CART',
        'NB': 'NB',
        'RF': 'RF',
        'XGBoost': 'XGBoost',
        'MLP': 'MLP'
    }
    
    acronym = acronym_map.get(model_name, model_name)

    result = {
        'model': model_name,
        'precision': None, 'precision_std': None,
        'recall': None, 'recall_std': None,
        'f1': None, 'f1_std': None,
        # Combined low-frequency (0-4) metrics
        'low_total': 0,
        'low_disagreements': 0,
        'low_disagreement_pct': 0.0,
        'low_safer': 0,
        'low_riskier': 0,
        'low_net': 0,
        # Global safety (for reference)
        'global_disagreements': 0,
        'global_safer': 0,
        'global_riskier': 0,
        'global_net': 0
    }

    # ── Performance metrics (global or aggregated) ───────────────────────────
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

    # ── Low-frequency safety: Extract Zero (0) and Very Low (1-4) separately ──
    
    # Pattern for Zero (0) section with dynamic acronym
    zero_pattern = rf'Zero \(0\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?{acronym}\s+Safer:\s*([0-9,]+).*?{acronym}\s+Riskier:\s*([0-9,]+)'
    
    # Pattern for Very Low (1-4) section with dynamic acronym
    verylow_pattern = rf'Very Low \(1-4\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?{acronym}\s+Safer:\s*([0-9,]+).*?{acronym}\s+Riskier:\s*([0-9,]+)'

    # Extract Zero (0) data
    zero_match = re.search(zero_pattern, content, re.DOTALL)
    if zero_match:
        zero_total = int(zero_match.group(1).replace(',', ''))
        zero_disagreements = int(zero_match.group(2).replace(',', ''))
        zero_safer = int(zero_match.group(3).replace(',', ''))
        zero_riskier = int(zero_match.group(4).replace(',', ''))
        
        result['low_total'] += zero_total
        result['low_disagreements'] += zero_disagreements
        result['low_safer'] += zero_safer
        result['low_riskier'] += zero_riskier

    # Extract Very Low (1-4) data
    verylow_match = re.search(verylow_pattern, content, re.DOTALL)
    if verylow_match:
        verylow_total = int(verylow_match.group(1).replace(',', ''))
        verylow_disagreements = int(verylow_match.group(2).replace(',', ''))
        verylow_safer = int(verylow_match.group(3).replace(',', ''))
        verylow_riskier = int(verylow_match.group(4).replace(',', ''))
        
        result['low_total'] += verylow_total
        result['low_disagreements'] += verylow_disagreements
        result['low_safer'] += verylow_safer
        result['low_riskier'] += verylow_riskier

    # Calculate net (safer - riskier)
    result['low_net'] = result['low_safer'] - result['low_riskier']

    # Compute Disagreement % for low-frequency
    if result['low_total'] > 0:
        result['low_disagreement_pct'] = (result['low_disagreements'] / result['low_total']) * 100
    else:
        result['low_disagreement_pct'] = 0.0

    # ── Global safety summary with dynamic acronym ───────────────────────────
    global_pattern = rf'GLOBAL SAFETY SUMMARY.*?Total disagreements:\s*([0-9,]+).*?{acronym}\s+Safer:\s*([0-9,]+).*?{acronym}\s+Riskier:\s*([0-9,]+).*?Net:\s*([+-]?\d+)'

    global_match = re.search(global_pattern, content, re.DOTALL)
    if global_match:
        result['global_disagreements'] = int(global_match.group(1).replace(',', ''))
        result['global_safer'] = int(global_match.group(2).replace(',', ''))
        result['global_riskier'] = int(global_match.group(3).replace(',', ''))
        result['global_net'] = int(global_match.group(4))

    return result

def format_pct(value):
    return f"{value:.1f}%" if value is not None else "N/A"

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

    # ── Print the 3-panel Markdown table ─────────────────────────────────────
    print("\n" + "="*70)
    print("Safety Analysis - Low-Frequency Regimes (count 0-4)")
    print("="*70)

    for pct in percentages:
        print(f"\nTraining percentage {pct}%")
        print("-"*60)
        print(f"{'Model':<12} {'Disagreement%':>14} {'Safer%':>10} {'Riskier%':>10} {'Net':>8}")
        print("-"*60)

        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                disag_pct = format_pct(r['low_disagreement_pct'])
                # CORRECTED: Divide by total samples, not disagreements
                safer_pct = format_pct((r['low_safer'] / r['low_total'] * 100) if r['low_total'] > 0 else 0)
                riskier_pct = format_pct((r['low_riskier'] / r['low_total'] * 100) if r['low_total'] > 0 else 0)
                net = f"{r['low_net']:+d}" if r['low_net'] is not None else "N/A"

                print(f"{display_name:<12} {disag_pct:>14} {safer_pct:>10} {riskier_pct:>10} {net:>8}")

        print("-"*60)

    # ── LaTeX table ─────────────────────────────────────────────────────────
    latex_output = []
    latex_output.append("\\begin{table}[!htb]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Safety analysis in low-frequency regimes (count 0-4) across training percentages.}")
    latex_output.append("\\label{tab:safety_low_freq}")
    latex_output.append("\\small")
    latex_output.append("\\begin{tabular}{lcccc}")
    latex_output.append("\\toprule")

    for pct in percentages:
        latex_output.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{Training {pct}\\%}}}} \\\\")
        latex_output.append("\\midrule")
        latex_output.append("Model & Disagreement \\% & Safer \\% & Riskier \\% & Net \\\\")
        latex_output.append("\\midrule")

        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                disag = format_pct(r['low_disagreement_pct'])
                # CORRECTED: Divide by total samples, not disagreements
                safer = format_pct((r['low_safer'] / r['low_total'] * 100) if r['low_total'] > 0 else 0)
                riskier = format_pct((r['low_riskier'] / r['low_total'] * 100) if r['low_total'] > 0 else 0)
                net = f"{r['low_net']:+d}" if r['low_net'] is not None else "N/A"
                latex_output.append(f"{display_name} & {disag} & {safer} & {riskier} & {net} \\\\")

        latex_output.append("\\midrule")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    with open("safety_low_freq_table.tex", "w") as f:
        f.write("\n".join(latex_output))

    # CSV
    csv_rows = []
    for pct in percentages:
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                csv_rows.append({
                    'percentage': pct,
                    'model': display_name,
                    'low_disagreement_pct': r['low_disagreement_pct'],
                    # CORRECTED: Divide by total samples, not disagreements
                    'low_safer_pct': (r['low_safer'] / r['low_total'] * 100) if r['low_total'] > 0 else 0,
                    'low_riskier_pct': (r['low_riskier'] / r['low_total'] * 100) if r['low_total'] > 0 else 0,
                    'low_net': r['low_net']
                })

    df = pd.DataFrame(csv_rows)
    df.to_csv("safety_low_freq_summary.csv", index=False)

    print("\nMarkdown table printed above.")
    print("LaTeX table saved to: safety_low_freq_table.tex")
    print("CSV summary saved to: safety_low_freq_summary.csv")

if __name__ == "__main__":
    main()
