#!/usr/bin/env python3
import re
import os
import pandas as pd
from pathlib import Path

def parse_testing_file(file_path):
    """Parse the testing_numeralia.txt file and extract metrics"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Use more flexible regex patterns to handle variations
        patterns = {
            'precision_avg': r'Average Precision:\s*([0-9.]+)',
            'precision_std': r'std dev Precision:\s*([0-9.]+)',
            'recall_avg': r'Average Recall:\s*([0-9.]+)',
            'recall_std': r'std dev recall:\s*([0-9.]+)',
            'f1_avg': r'Average F1-score:\s*([0-9.]+)',
            'f1_std': r'std dev F1-score:\s*([0-9.]+)',
            'accuracy_avg': r'Average Accuracy:\s*([0-9.]+)',
            'accuracy_std': r'std dev Accuracy:\s*([0-9.]+)',
            'time_avg': r'Avg\. testing time:\s*([0-9.]+)',
            'time_std': r'Stdev\. testing time:\s*([0-9.]+)'
        }
        
        # Alternative patterns if the first ones don't match
        alt_patterns = {
            'recall_std': r'std dev Recall:\s*([0-9.]+)',
            'f1_std': r'std dev F1:\s*([0-9.]+)',
            'precision_std': r'std dev Prec:\s*([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if not match and key in alt_patterns:
                match = re.search(alt_patterns[key], content)
            
            if match:
                metrics[key] = float(match.group(1))
            else:
                metrics[key] = None
                
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    return metrics

def format_metric(avg, std):
    """Format metric with average and standard deviation"""
    if avg is None or std is None:
        return "N/A"
    return f"{avg:.4f} ± {std:.4f}"

def main():
    # Define the file paths - now including Train_100
    file_paths = [
        # Train_01 files
        "../Train_01/models/RF/Results/testing_numeralia.txt",
        "../Train_01/models/XGBoost/Results/testing_numeralia.txt",
        "../Train_01/models/PL-fMDP/Results/testing_numeralia.txt",
        "../Train_01/models/CART/Results/testing_numeralia.txt",
        "../Train_01/models/NB/Results/testing_numeralia.txt",
        "../Train_01/models/MLP/Results/testing_numeralia.txt",
        "../Train_01/models/LR/Results/testing_numeralia.txt",
        
        # Train_50 files  
        "../Train_50/models/RF/Results/testing_numeralia.txt",
        "../Train_50/models/XGBoost/Results/testing_numeralia.txt",
        "../Train_50/models/PL-fMDP/Results/testing_numeralia.txt",
        "../Train_50/models/CART/Results/testing_numeralia.txt",
        "../Train_50/models/NB/Results/testing_numeralia.txt",
        "../Train_50/models/MLP/Results/testing_numeralia.txt",
        "../Train_50/models/LR/Results/testing_numeralia.txt",
        
        # Train_100 files - ADDED THESE
        "../Train_100/models/RF/Results/testing_numeralia.txt",
        "../Train_100/models/XGBoost/Results/testing_numeralia.txt",
        "../Train_100/models/PL-fMDP/Results/testing_numeralia.txt",
        "../Train_100/models/CART/Results/testing_numeralia.txt",
        "../Train_100/models/NB/Results/testing_numeralia.txt",
        "../Train_100/models/MLP/Results/testing_numeralia.txt",
        "../Train_100/models/LR/Results/testing_numeralia.txt",
    ]
    
    # Extract model names and training percentages from file paths
    data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist - {file_path}")
            continue
            
        path = Path(file_path)
        model_name = path.parent.parent.name
        training_percent = path.parent.parent.parent.parent.name.replace('Train_', '')
        
        # Fix model name formatting
        if model_name == "PL-fMDP":
            model_name = "PL-fMDPs"
        elif model_name == "RF":
            model_name = "RFs"
        
        metrics = parse_testing_file(file_path)
        if metrics:
            metrics['model'] = model_name
            metrics['training_percent'] = training_percent
            data.append(metrics)
            
            # Print the parsed data for verification
            print(f"\nParsed {file_path}:")
            print(f"  Model: {model_name}, Training %: {training_percent}")
            print(f"  Precision: {format_metric(metrics.get('precision_avg'), metrics.get('precision_std'))}")
            print(f"  Recall: {format_metric(metrics.get('recall_avg'), metrics.get('recall_std'))}")
            print(f"  F1-score: {format_metric(metrics.get('f1_avg'), metrics.get('f1_std'))}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Define the desired model order
    model_order = ['PL-fMDPs', 'LR', 'CART', 'NB', 'RFs', 'XGBoost', 'MLP']
    
    # Generate LaTeX table with all three training percentages
    latex_output = []
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Model Performance Comparison Across Different Training Percentages}")
    latex_output.append("\\label{tab:model_comparison}")
    latex_output.append("\\begin{tabular}{|l|ccc|}")
    latex_output.append("\\hline")
    
    # Training percentage 1%
    latex_output.append("\\multicolumn{4}{|c|}{\\textbf{Training percentage 1\\%}} \\\\")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Models} & \\textbf{Precision (Avg±std)} & \\textbf{Recall (Avg±std)} & \\textbf{F1-score (Avg±std)} \\\\")
    latex_output.append("\\hline")
    
    for model in model_order:
        data_1 = df[(df['model'] == model) & (df['training_percent'] == '01')]
        if len(data_1) > 0:
            row = f"{model} & {format_metric(data_1['precision_avg'].iloc[0], data_1['precision_std'].iloc[0])} & {format_metric(data_1['recall_avg'].iloc[0], data_1['recall_std'].iloc[0])} & {format_metric(data_1['f1_avg'].iloc[0], data_1['f1_std'].iloc[0])} \\\\"
        else:
            row = f"{model} & N/A & N/A & N/A \\\\"
        latex_output.append(row)
    
    latex_output.append("\\hline")
    
    # Training percentage 50%
    latex_output.append("\\multicolumn{4}{|c|}{\\textbf{Training percentage 50\\%}} \\\\")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Models} & \\textbf{Precision (Avg±std)} & \\textbf{Recall (Avg±std)} & \\textbf{F1-score (Avg±std)} \\\\")
    latex_output.append("\\hline")
    
    for model in model_order:
        data_50 = df[(df['model'] == model) & (df['training_percent'] == '50')]
        if len(data_50) > 0:
            row = f"{model} & {format_metric(data_50['precision_avg'].iloc[0], data_50['precision_std'].iloc[0])} & {format_metric(data_50['recall_avg'].iloc[0], data_50['recall_std'].iloc[0])} & {format_metric(data_50['f1_avg'].iloc[0], data_50['f1_std'].iloc[0])} \\\\"
        else:
            row = f"{model} & N/A & N/A & N/A \\\\"
        latex_output.append(row)
    
    latex_output.append("\\hline")
    
    # Training percentage 100% - ADDED THIS SECTION
    latex_output.append("\\multicolumn{4}{|c|}{\\textbf{Training percentage 100\\%}} \\\\")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Models} & \\textbf{Precision (Avg±std)} & \\textbf{Recall (Avg±std)} & \\textbf{F1-score (Avg±std)} \\\\")
    latex_output.append("\\hline")
    
    for model in model_order:
        data_100 = df[(df['model'] == model) & (df['training_percent'] == '100')]
        if len(data_100) > 0:
            row = f"{model} & {format_metric(data_100['precision_avg'].iloc[0], data_100['precision_std'].iloc[0])} & {format_metric(data_100['recall_avg'].iloc[0], data_100['recall_std'].iloc[0])} & {format_metric(data_100['f1_avg'].iloc[0], data_100['f1_std'].iloc[0])} \\\\"
        else:
            row = f"{model} & N/A & N/A & N/A \\\\"
        latex_output.append(row)
    
    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Print LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE OUTPUT:")
    print("="*80)
    for line in latex_output:
        print(line)
    
    # Save LaTeX table to file
    with open('model_comparison_table_1.tex', 'w') as f:
        f.write("\n".join(latex_output))
    print("\nLaTeX table saved to 'model_comparison_table_1.tex'")

if __name__ == "__main__":
    main()
