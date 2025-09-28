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
            param_cols = [col for col in df.columns if col not in ['fold', 'f1_score', 'F1_Score', 'validation_f1']]
            
            # Create string representations of parameter combinations
            combinations = []
            for _, row in df.iterrows():
                combo_str = []
                for col in param_cols:
                    value = row[col]
                    # Format values appropriately
                    if pd.isna(value):
                        continue
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    combo_str.append(f"{col}={value}")
                combinations.append(", ".join(combo_str))
            
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

def format_hyperparameter_name(param_name, value):
    """Format hyperparameter names and values according to the reference table"""
    param_mapping = {
        # PL-fMDP
        'gamma': 'Discount factor',
        'epsilon': 'Maximum error bound',
        
        # CART
        'max_depth': 'Maximum depth',
        'min_samples_split': 'Minimum samples to split',
        'min_samples_leaf': 'Minimum samples in leaf',
        
        # XGBoost
        'max_depth': 'Maximum depth',
        'learning_rate': 'Learning rate',
        'n_estimators': 'Number of estimators',
        'colsample_bytree': 'Subsample ratio of columns',
        'subsample': 'Subsample ratio',
        
        # RF
        'max_depth': 'Maximum depth',
        'n_estimators': 'Number of estimators',
        'max_features': 'Maximum features for split',
        'min_samples_split': 'Minimum samples to split',
        
        # NB
        'alpha': 'Additive smoothing',
        'fit_prior': 'Learning of priors',
        
        # MLP
        'hidden_layer_sizes': 'Size of hidden layers',
        'activation': 'Activation function',
        'solver': 'Solver',
        'max_iter': 'Maximum iterations',
        
        # LR
        'C': 'Inverse regularization strength',
        'penalty': 'Penalty',
        'class_weight': 'Class weight',
        'Class_Weight': 'Class weight',
        'Penalty': 'Penalty'
    }
    
    # Format value appropriately
    if isinstance(value, str):
        formatted_value = value
    elif isinstance(value, float) and value.is_integer():
        formatted_value = str(int(value))
    else:
        formatted_value = str(value)
    
    # Handle special cases
    if param_name == 'hidden_layer_sizes' and formatted_value.startswith('('):
        formatted_value = formatted_value.replace('(', '').replace(')', '').replace(',', '×')
    
    if param_name == 'max_features':
        if formatted_value == "'sqrt'":
            formatted_value = 'sqrt'
        elif formatted_value == "'log2'":
            formatted_value = 'log2'
    
    if param_name in ['class_weight', 'Class_Weight']:
        if formatted_value == 'None':
            formatted_value = 'None'
        elif formatted_value == "'balanced'":
            formatted_value = 'balanced'
    
    if param_name == 'fit_prior':
        formatted_value = 'True' if str(value).lower() == 'true' else 'False'
    
    # Get the formatted parameter name
    formatted_param = param_mapping.get(param_name, param_name)
    
    return f"{formatted_param}={formatted_value}"

def generate_latex_table(results):
    latex_code = """\\begin{sidewaystable}[!tb]
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
        model_printed = False
        
        for percentage in percentages_order:
            if model in results[percentage]:
                data = results[percentage][model]
                
                # Parse and format the hyperparameter combination
                combo_parts = data['combination'].split(', ')
                formatted_combo = []
                
                for part in combo_parts:
                    if '=' in part:
                        param, value = part.split('=', 1)
                        # Clean up value formatting
                        value = value.strip()
                        if value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        formatted_part = format_hyperparameter_name(param, value)
                        formatted_combo.append(formatted_part)
                
                combo_str = ", ".join(formatted_combo)
                
                # Format percentage display
                display_percentage = f"{int(percentage)}\\%" if percentage != '01' else "1\\%"
                freq_str = f"{data['frequency']}/{data['total_folds']}"
                
                if first_row:
                    latex_code += f"\\multirow{{3}}{{*}}{{{model}}} & {display_percentage} & {combo_str} & {freq_str} \\\\\n"
                    first_row = False
                    model_printed = True
                else:
                    latex_code += f" & {display_percentage} & {combo_str} & {freq_str} \\\\\n"
        
        if model_printed:
            latex_code += "\\midrule\n"
    
    latex_code += """\\bottomrule
\\end{tabular}}
\\label{tab:hyperparameter_frequency}
\\end{sidewaystable}"""
    
    return latex_code

def main():
    results = analyze_hyperparameters()
    
    # Debug: print raw results to check frequencies
    print("Raw results for verification:")
    for percentage in ['01', '50', '100']:
        print(f"\nPercentage {percentage}:")
        for model in results[percentage]:
            data = results[percentage][model]
            print(f"  {model}: {data['combination']} - {data['frequency']}/{data['total_folds']}")
    
    latex_table = generate_latex_table(results)
    
    # Save to file
    with open("hyperparameter_frequency_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\nLaTeX table generated and saved to hyperparameter_frequency_table.tex")
    print("\nGenerated table preview:")
    print(latex_table[:500] + "...")  # Show first 500 characters

if __name__ == "__main__":
    main()
