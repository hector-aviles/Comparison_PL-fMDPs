import pandas as pd
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hyperparameter_distribution():
    percentages = ['01', '50', '100']
    models = ['RF', 'XGBoost', 'PL-fMDP', 'CART', 'NB', 'MLP', 'LR']
    
    all_results = {}
    
    for percentage in percentages:
        all_results[percentage] = {}
        for model in models:
            file_path = f"../Train_{percentage}/models/{model}/best_parameters.csv"
            if not os.path.exists(file_path):
                continue
                
            df = pd.read_csv(file_path)
            all_results[percentage][model] = df
    
    return all_results

def create_hyperparameter_summary_table(all_results):
    """Create a compact summary table showing most frequent values for each hyperparameter"""
    
    summary_data = []
    
    for model in ['PL-fMDP', 'CART', 'XGBoost', 'RF', 'NB', 'MLP', 'LR']:
        model_data = {'Model': model}
        
        for percentage in ['01', '50', '100']:
            if model not in all_results[percentage]:
                continue
                
            df = all_results[percentage][model]
            
            # Define non-hyperparameter columns for each model
            non_param_cols = ['fold']
            if model == 'LR':
                non_param_cols.extend(['f1_score', 'F1_Score'])
            elif model == 'MLP':
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            else:
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            
            param_cols = [col for col in df.columns if col not in non_param_cols]
            
            param_summary = []
            for param in param_cols:
                # Get most frequent value for this parameter
                value_counts = df[param].value_counts()
                if len(value_counts) > 0:
                    most_common = value_counts.index[0]
                    frequency = value_counts.iloc[0]
                    total = len(df)
                    
                    # Format the value
                    if isinstance(most_common, float) and most_common.is_integer():
                        most_common = int(most_common)
                    
                    param_summary.append(f"{param}={most_common}({frequency}/{total})")
            
            model_data[f"{int(percentage)}%"] = "; ".join(param_summary)
        
        summary_data.append(model_data)
    
    return pd.DataFrame(summary_data)

def get_parameter_acronym(param_name, model):
    """Convert parameter names to acronyms for better readability"""
    acronyms = {
        # MLP acronyms
        'max_iter': 'max_it',
        'hidden_layer_sizes': 'h_l_s',
        
        # RF acronyms
        'n_estimators': 'n_est',
        'max_depth': 'max_d',
        'min_samples_split': 'min_s_s',
        'min_samples_leaf': 'min_s_l',
        'max_features': 'max_feat',
        
        # XGBoost acronyms
        'learning_rate': 'l_r',
        'subsample': 'ss',
        'colsample_bytree': 'cs_bt',
        
        # CART acronyms (same as RF for consistency)
        'min_samples_split': 'min_s_s',
        'min_samples_leaf': 'min_s_l',
        'max_depth': 'max_d'
    }
    
    return acronyms.get(param_name, param_name)

def create_hyperparameter_heatmaps(all_results):
    """Create heatmaps showing parameter value distributions with improved layout"""
    
    # Define fixed figure size for consistency (width, height in inches)
    fig_width, fig_height = 18, 6
    
    for model in ['RF', 'XGBoost', 'PL-fMDP', 'CART', 'NB', 'MLP', 'LR']:
        # Create figure with fixed size and constrained layout
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), 
                                gridspec_kw={'width_ratios': [1, 1, 1.1]})  # Slightly wider last column for colorbar
        
        # Create a single colorbar for the entire figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        
        # Set sizes
        annot_kws = {'size': 16, 'weight': 'bold'}
        x_ticksize = 14
        y_ticksize = 16  # bumped +2 units for hyperparameter names
        cbar_ticksize = 16  # bumped +2 units for frequency labels
        
        for i, percentage in enumerate(['01', '50', '100']):
            if model not in all_results[percentage]:
                axes[i].set_visible(False)
                continue
                
            df = all_results[percentage][model]
            
            # Define non-hyperparameter columns for each model
            non_param_cols = ['fold']
            if model == 'LR':
                non_param_cols.extend(['f1_score', 'F1_Score'])
            elif model == 'MLP':
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            else:
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            
            param_cols = [col for col in df.columns if col not in non_param_cols]
            
            # Create value count matrix for heatmap
            value_data = []
            for param in param_cols:
                value_counts = df[param].value_counts().head(3)
                                
                for value, count in value_counts.items():
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                    value_data.append([param, str(value), count])
            
            if value_data:
                heatmap_df = pd.DataFrame(value_data, columns=['Parameter', 'Value', 'Count'])
                pivot_df = heatmap_df.pivot(index='Parameter', columns='Value', values='Count').fillna(0)
                # Replace parameter names with acronyms for y-axis labels
                pivot_df.index = [get_parameter_acronym(param, model) for param in pivot_df.index]
                
                # Verify row sums
                row_sums = pivot_df.sum(axis=1)
                for param, row_sum in row_sums.items():
                    if row_sum != 20:
                        print(f"Warning: {model} {percentage}% {param} sums to {row_sum}, not 20")
                
                # Create heatmap with improved layout
                if i == 2:  # Last subplot gets the colorbar
                    heatmap = sns.heatmap(pivot_df, 
                                       annot=True, 
                                       fmt='.0f', 
                                       cmap='YlOrRd', 
                                       ax=axes[i], 
                                       cbar=True,
                                       cbar_ax=cbar_ax,
                                       cbar_kws={'label': 'Frequency'},
                                       annot_kws=annot_kws)
                else:  # First two subplots have no colorbar
                    heatmap = sns.heatmap(pivot_df, 
                                       annot=True, 
                                       fmt='.0f', 
                                       cmap='YlOrRd', 
                                       ax=axes[i], 
                                       cbar=False,
                                       annot_kws=annot_kws)
                
                axes[i].set_title(f'{int(percentage)}% Training data', fontsize=17)
                axes[i].tick_params(axis='x', rotation=0, labelsize=x_ticksize)
                axes[i].tick_params(axis='y', rotation=90, labelsize=y_ticksize)
                axes[i].set_ylabel('')
                axes[i].set_xlabel('Value', fontsize=16)
                axes[i].set_aspect('auto')
                
                # Remove y-axis labels for all but the first subplot
                if i > 0:
                    axes[i].set_yticklabels([])
                    axes[i].set_ylabel('')
        
        # Set the colorbar label and ticks
        cbar_ax.set_ylabel('Frequency', rotation=270, labelpad=15, fontsize=16)
        cbar_ax.tick_params(labelsize=cbar_ticksize)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0.05, 0, 0.90, 0.95])  # Leave space for the colorbar on the right
        plt.savefig(f'{model}_hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_parameter_stability_chart(all_results):
    """Chart showing how parameter values change across percentages"""
    
    stable_params = []
    changing_params = []
    
    for model in ['PL-fMDP', 'CART', 'XGBoost', 'RF', 'NB', 'MLP', 'LR']:
        param_values = {}
        
        for percentage in ['01', '50', '100']:
            if model not in all_results[percentage]:
                continue
                
            df = all_results[percentage][model]
            
            # Define non-hyperparameter columns for each model
            non_param_cols = ['fold']
            if model == 'LR':
                non_param_cols.extend(['f1_score', 'F1_Score'])
            elif model == 'MLP':
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            else:
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            
            param_cols = [col for col in df.columns if col not in non_param_cols]
            
            for param in param_cols:
                most_common = df[param].mode()[0] if len(df[param].mode()) > 0 else None
                if most_common is not None:
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append((percentage, most_common))
        
        # Check stability
        for param, values in param_values.items():
            if len(values) == 3:
                unique_values = len(set(value[1] for value in values))
                if unique_values == 1:
                    stable_params.append((model, param, values[0][1]))
                else:
                    changing_params.append((model, param, [f"{p}:{v}" for p, v in values]))
    
    print("Stable Hyperparameters (same value across all percentages):")
    for model, param, value in stable_params:
        print(f"  {model}: {param} = {value}")
    
    print("\nChanging Hyperparameters:")
    for model, param, values in changing_params:
        print(f"  {model}: {param} = {', '.join(values)}")

def generate_detailed_latex_table(all_results):
    """Generate a detailed LaTeX table with hyperparameter information"""
    
    latex_code = """\\begin{sidewaystable}[!tb]
\\centering
\\caption{Hyperparameter value distributions across different training set percentages}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lccc}
\\toprule
Model & \\makecell{1\\% Training} & \\makecell{50\\% Training} & \\makecell{100\\% Training} \\\\
\\midrule
"""
    
    for model in ['PL-fMDP', 'CART', 'XGBoost', 'RF', 'NB', 'MLP', 'LR']:
        model_row = f"{model} & "
        percentages_data = []
        
        for percentage in ['01', '50', '100']:
            if model not in all_results[percentage]:
                percentages_data.append("--")
                continue
                
            df = all_results[percentage][model]
            
            # Define non-hyperparameter columns for each model
            non_param_cols = ['fold']
            if model == 'LR':
                non_param_cols.extend(['f1_score', 'F1_Score'])
            elif model == 'MLP':
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            else:
                non_param_cols.extend(['f1_score', 'F1_Score', 'validation_f1'])
            
            param_cols = [col for col in df.columns if col not in non_param_cols]
            
            param_info = []
            for param in param_cols:
                value_counts = df[param].value_counts().head(2)
                
                if len(value_counts) > 0:
                    top_value = value_counts.index[0]
                    top_count = value_counts.iloc[0]
                    
                    if isinstance(top_value, float) and top_value.is_integer():
                        top_value = int(top_value)
                    
                    if len(value_counts) == 1 or value_counts.iloc[0] > value_counts.iloc[1] * 2:
                        param_info.append(f"\\textbf{{{param}={top_value}}} ({top_count}/20)")
                    else:
                        second_value = value_counts.index[1]
                        second_count = value_counts.iloc[1]
                        param_info.append(f"{param}={top_value}({top_count}),{second_value}({second_count})")
            
            percentages_data.append("; ".join(param_info))
        
        model_row += " & ".join(percentages_data) + " \\\\\n\\midrule\n"
        latex_code += model_row
    
    latex_code += """\\bottomrule
\\end{tabular}}
\\label{tab:hyperparameter_distribution}
\\end{sidewaystable}"""
    
    with open("detailed_hyperparameter_table.tex", "w") as f:
        f.write(latex_code)
    
    return latex_code

def main():
    print("Analyzing hyperparameter data from best_parameters.csv files...")
    
    # Load all data
    all_results = analyze_hyperparameter_distribution()
    
    # Create summary table
    summary_df = create_hyperparameter_summary_table(all_results)
    print("\nHyperparameter Summary Table:")
    print(summary_df.to_string(index=False))
    
    # Generate visualizations
    print("\nCreating heatmaps...")
    create_hyperparameter_heatmaps(all_results)
    
    # Analyze parameter stability
    print("\nAnalyzing parameter stability across percentages...")
    create_parameter_stability_chart(all_results)
    
    # Generate detailed LaTeX table
    print("\nGenerating detailed LaTeX table...")
    latex_table = generate_detailed_latex_table(all_results)
    
    print("\nAnalysis complete! Files generated:")
    print("- detailed_hyperparameter_table.tex")
    print("- Heatmap images for each model")
    print("- Summary printed to console")

if __name__ == "__main__":
    main()

'''
def get_acronym_explanation(model):
    """Generate acronym explanations for each model"""
    explanations = {
        'MLP': 'Acronyms: max_iter (max_it), hidden_layer_sizes (h_l_s)',
        'RF': 'Acronyms: n_estimators (n_est), max_depth (max_d), min_samples_split (min_s_s), min_samples_leaf (min_s_l), max_features (max_feat)',
        'XGBoost': 'Acronyms: max_depth (max_d), learning_rate (learn_rate), n_estimators (n_est), subsample (ss), colsample_bytree (cs_bt)',
        'CART': 'Acronyms: min_samples_split (min_s_s), min_samples_leaf (min_s_l), max_depth (max_d)',
        'PL-fMDP': '',  # No acronyms needed
        'NB': '',       # No acronyms needed
        'LR': ''        # No acronyms needed
    }
    return explanations.get(model, '')
'''

