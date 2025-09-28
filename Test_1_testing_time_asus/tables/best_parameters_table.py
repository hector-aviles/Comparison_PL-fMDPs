import pandas as pd
import os
import glob
from collections import Counter
import numpy as np

def analyze_hyperparameters():
    percentages = ['01', '50', '100']
    models = ['RF', 'XGBoost', 'PL-fMDP', 'CART', 'NB', 'MLP', 'LR']
    
    results = {}
    
    for percentage in percentages:
        results[percentage] = {}
        for model in models:
            file_path = f"../Train_{percentage}/models/{model}/best_parameters.csv"
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            # Remove fold and score columns for combination analysis
            param_cols = [col for col in df.columns if col not in ['fold', 'f1_score']]
            
            # Create string representations of parameter combinations
            combinations = []
            for _, row in df.iterrows():
                combo_str = []
                for col in param_cols:
                    combo_str.append(f"{col}={row[col]}")
                combinations.append("; ".join(combo_str))
            
            # Count frequency of each combination
            combo_counts = Counter(combinations)
            most_common = combo_counts.most_common(1)
            
            if most_common:
                most_common_combo, frequency = most_common[0]
                results[percentage][model] = {
                    'combination': most_common_combo,
                    'frequency': frequency,
                    'total_folds': len(df)
                }
    
    return results

def generate_latex_table(results):
    latex_code = """\\begin{table}[!tb]
\\centering
\\caption{Most frequent hyperparameter combinations for each model across different training set percentages.}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccc}
\\toprule
 & \\makecell{Training \\\\ percentage} & \\makecell{Most frequent \\\\ hyperparameter combination} & \\makecell{Frequency \\\\(out of 20 folds)} \\\\
\\midrule
"""

    models_order = ['PL-fMDP', 'CART', 'XGBoost', 'RF', 'NB', 'MLP', 'LR']
    percentages_order = ['01', '50', '100']
    
    for model in models_order:
        first_row = True
        for percentage in percentages_order:
            if model in results[percentage]:
                data = results[percentage][model]
                combo_parts = data['combination'].split('; ')
                formatted_combo = []
                
                for part in combo_parts:
                    if '=' in part:
                        param, value = part.split('=', 1)
                        # Format specific parameters
                        if param in ['max_depth', 'min_samples_split', 'min_samples_leaf', 
                                   'n_estimators', 'max_features', 'alpha', 'hidden_layer_sizes',
                                   'max_iter', 'C']:
                            formatted_combo.append(f"{param}={value}")
                        else:
                            formatted_combo.append(part)
                
                combo_str = "; ".join(formatted_combo)
                freq_str = f"{data['frequency']}/{data['total_folds']}"
                
                if first_row:
                    latex_code += f"\\multirow{{3}}{{*}}{{{model}}} & {percentage}\\% & {combo_str} & {freq_str} \\\\\n"
                    first_row = False
                else:
                    latex_code += f" & {percentage}\\% & {combo_str} & {freq_str} \\\\\n"
        
        latex_code += "\\midrule\n"
    
    latex_code += """\\bottomrule
\\end{tabular}}
\\label{tab:hyperparameter_frequency}
\\end{table}"""
    
    return latex_code

def main():
    results = analyze_hyperparameters()
    latex_table = generate_latex_table(results)
    
    # Save to file
    with open("hyperparameter_frequency_table.tex", "w") as f:
        f.write(latex_table)
    
    print("LaTeX table generated and saved to hyperparameter_frequency_table.tex")
    print("\nGenerated table:")
    print(latex_table)

if __name__ == "__main__":
    main()
