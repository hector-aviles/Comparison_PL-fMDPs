import os
import re
import pandas as pd

def parse_metrics_file(filepath, model_name):
    """
    Parse a model's testing frequency file to extract:
    Safety analysis for combined Low-Frequency (0-4) category
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    result = {
        'model': model_name,
        'low_safer': 0, 
        'low_riskier': 0, 
        'low_net': 0
    }
    
    # Try to find combined Low-Frequency (0-4) section first
    low_combined_pattern = r'Low \(0-4\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                          r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Safer:\s*([0-9,]+).*?' + \
                          r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Riskier:\s*([0-9,]+).*?' + \
                          r'Net safety improvement:\s*([+-]?\d+)'
    
    low_match = re.search(low_combined_pattern, content, re.DOTALL)
    if low_match:
        low_safer = int(low_match.group(3).replace(',', ''))
        low_riskier = int(low_match.group(4).replace(',', ''))
        low_net = int(low_match.group(5))
        
        result['low_safer'] = low_safer
        result['low_riskier'] = low_riskier
        result['low_net'] = low_net
        print(f"  Found combined Low (0-4) for {model_name}: Safer={low_safer}, Riskier={low_riskier}, Net={low_net}")
    else:
        # If not combined, try to extract Zero and Very Low separately and combine them
        # Extract Zero (0) category
        zero_pattern = r'Zero \(0\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                      r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Safer:\s*([0-9,]+).*?' + \
                      r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Riskier:\s*([0-9,]+).*?' + \
                      r'Net safety improvement:\s*([+-]?\d+)'
        
        zero_match = re.search(zero_pattern, content, re.DOTALL)
        
        # Extract Very Low (1-4) category
        verylow_pattern = r'Very Low \(1-4\):.*?Total samples:\s*([0-9,]+).*?Total disagreements:\s*([0-9,]+).*?' + \
                         r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Safer:\s*([0-9,]+).*?' + \
                         r'(?:AP|CART|NB|LR|RF|XGBoost|MLP|XGBoost)\s+Riskier:\s*([0-9,]+).*?' + \
                         r'Net safety improvement:\s*([+-]?\d+)'
        
        verylow_match = re.search(verylow_pattern, content, re.DOTALL)
        
        if zero_match and verylow_match:
            zero_safer = int(zero_match.group(3).replace(',', ''))
            zero_riskier = int(zero_match.group(4).replace(',', ''))
            zero_net = int(zero_match.group(5))
            
            verylow_safer = int(verylow_match.group(3).replace(',', ''))
            verylow_riskier = int(verylow_match.group(4).replace(',', ''))
            verylow_net = int(verylow_match.group(5))
            
            result['low_safer'] = zero_safer + verylow_safer
            result['low_riskier'] = zero_riskier + verylow_riskier
            result['low_net'] = zero_net + verylow_net
            
            print(f"  Combined Zero+VeryLow for {model_name}: Safer={result['low_safer']}, Riskier={result['low_riskier']}, Net={result['low_net']}")
        elif zero_match:
            # Only Zero found
            result['low_safer'] = int(zero_match.group(3).replace(',', ''))
            result['low_riskier'] = int(zero_match.group(4).replace(',', ''))
            result['low_net'] = int(zero_match.group(5))
            print(f"  Only Zero found for {model_name}: Safer={result['low_safer']}, Riskier={result['low_riskier']}, Net={result['low_net']}")
        elif verylow_match:
            # Only Very Low found
            result['low_safer'] = int(verylow_match.group(3).replace(',', ''))
            result['low_riskier'] = int(verylow_match.group(4).replace(',', ''))
            result['low_net'] = int(verylow_match.group(5))
            print(f"  Only Very Low found for {model_name}: Safer={result['low_safer']}, Riskier={result['low_riskier']}, Net={result['low_net']}")
    
    return result

def calculate_net_percentage(safer, riskier):
    """Calculate net percentage as Safer% - Riskier%"""
    total = safer + riskier
    if total == 0:
        return 0.0
    safer_pct = (safer / total * 100)
    riskier_pct = (riskier / total * 100)
    return safer_pct - riskier_pct

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
    
    print("\n" + "="*80)
    print("COLLECTING SAFETY METRICS FROM ALL MODELS")
    print("="*80)
    
    # Collect all results
    all_results = {}
    
    for pct in percentages:
        print(f"\nProcessing Training {pct}%:")
        print("-"*40)
        pct_results = {}
        
        for display_name, model_folder, filename in models:
            filepath = f"../Test_{pct}/models/{model_folder}/Results/{filename}"
            print(f"  Checking: {filepath}")
            result = parse_metrics_file(filepath, display_name)
            if result and (result['low_safer'] > 0 or result['low_riskier'] > 0):
                pct_results[display_name] = result
                print(f"    ✓ Added {display_name}")
            else:
                print(f"    ✗ No data for {display_name}")
        
        all_results[pct] = pct_results
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append("\\begin{table}[!htb]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Safety-oriented disagreement summary for in-distribution testing. Results combine zero-frequency and very-low-frequency states. Net (\\%) = Safer (\\%) - Riskier (\\%).}")
    latex_output.append("\\label{tab:safety_summary}")
    latex_output.append("\\small")
    latex_output.append("\\begin{tabular}{lccc}")
    latex_output.append("\\toprule")
    
    for pct in percentages:
        latex_output.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{Training {pct}\\%}}}} \\\\")
        latex_output.append("\\midrule")
        latex_output.append("Model & Safer (\\%) & Riskier (\\%) & Net (\\%) \\\\")
        latex_output.append("\\midrule")
        
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                
                # Calculate percentages
                total = r['low_safer'] + r['low_riskier']
                if total > 0:
                    safer_pct = (r['low_safer'] / total * 100)
                    riskier_pct = (r['low_riskier'] / total * 100)
                    net_pct = safer_pct - riskier_pct
                else:
                    safer_pct = riskier_pct = net_pct = 0.0
                
                row = f"{display_name} & {safer_pct:.1f}\\% & {riskier_pct:.1f}\\% & {net_pct:+.1f}\\% \\\\"
                latex_output.append(row)
            else:
                # Placeholder for missing data
                latex_output.append(f"{display_name} & --- & --- & --- \\\\")
        
        latex_output.append("\\midrule")
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}\\\\\text{Net (%)} = \text{Safer (%)} - \text{Riskier (%) }")
    latex_output.append("\\end{table}")
    
    # Write LaTeX table to file
    with open("safety_summary_table.tex", "w", encoding='utf-8') as f:
        f.write("\n".join(latex_output))
    
    # Also generate a CSV summary
    csv_rows = []
    for pct in percentages:
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                total = r['low_safer'] + r['low_riskier']
                if total > 0:
                    safer_pct = (r['low_safer'] / total * 100)
                    riskier_pct = (r['low_riskier'] / total * 100)
                    net_pct = safer_pct - riskier_pct
                else:
                    safer_pct = riskier_pct = net_pct = 0.0
                
                csv_rows.append({
                    'percentage': pct,
                    'model': display_name,
                    'safer_count': r['low_safer'],
                    'riskier_count': r['low_riskier'],
                    'safer_pct': f"{safer_pct:.1f}",
                    'riskier_pct': f"{riskier_pct:.1f}",
                    'net_pct': f"{net_pct:+.1f}"
                })
    
    df = pd.DataFrame(csv_rows)
    df.to_csv("safety_summary.csv", index=False)
    
    # Print summary to console
    print("\n" + "="*80)
    print("SAFETY-ORIENTED DISAGREEMENT SUMMARY - LOW-FREQUENCY STATES (0-4)")
    print("="*80)
    
    for pct in percentages:
        print(f"\nTraining {pct}%:")
        print("-"*70)
        print(f"{'Model':<10} {'Safer%':>8} {'Riskier%':>10} {'Net%':>8}")
        print("-"*70)
        
        for display_name, _, _ in models:
            if display_name in all_results[pct]:
                r = all_results[pct][display_name]
                total = r['low_safer'] + r['low_riskier']
                if total > 0:
                    safer_pct = (r['low_safer'] / total * 100)
                    riskier_pct = (r['low_riskier'] / total * 100)
                    net_pct = safer_pct - riskier_pct
                    print(f"{display_name:<10} {safer_pct:>7.1f}% {riskier_pct:>9.1f}% {net_pct:>+7.1f}%")
                else:
                    print(f"{display_name:<10} {'---':>7} {'---':>9} {'---':>7}")
            else:
                print(f"{display_name:<10} {'---':>7} {'---':>9} {'---':>7}")
    
    print("\n" + "="*80)
    print(f"LaTeX table saved to: safety_summary_table.tex")
    print(f"CSV summary saved to: safety_summary.csv")
    print("="*80)

if __name__ == "__main__":
    main()
