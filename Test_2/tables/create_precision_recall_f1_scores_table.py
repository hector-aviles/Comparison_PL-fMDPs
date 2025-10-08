#!/usr/bin/env python3
import re
import os
import pandas as pd
from pathlib import Path

def parse_testing_file(file_path):
    """Parse the testing_numeralia.txt file and extract driver-specific metrics"""
    driver_metrics = {1: {}, 2: {}, 3: {}, 4: {}}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        for driver in range(1, 5):
            # Parse metrics for all files (same format for PL-fMDP and others)
            patterns = {
                f'precision_avg_d{driver}': rf'Driver {driver} Metrics:.*?Precision:\s*([0-9.]+),\s*Std Dev:\s*([0-9.]+)',
                f'recall_avg_d{driver}': rf'Driver {driver} Metrics:.*?Recall:\s*([0-9.]+),\s*Std Dev:\s*([0-9.]+)',
                f'f1_avg_d{driver}': rf'Driver {driver} Metrics:.*?F1:\s*([0-9.]+),\s*Std Dev:\s*([0-9.]+)'
            }
            
            driver_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    metric_name = key.replace(f'_d{driver}', '')
                    if metric_name.endswith('_avg'):
                        driver_data[metric_name] = float(match.group(1))
                        driver_data[metric_name.replace('_avg', '_std')] = float(match.group(2))
                else:
                    driver_data[key.replace(f'_d{driver}', '')] = None
            
            driver_metrics[driver] = driver_data
                
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    return driver_metrics

def format_metric(avg, std):
    """Format metric with average and standard deviation"""
    if avg is None or std is None:
        return "N/A"
    return f"{avg:.4f} $\\pm$ {std:.4f}"

def main():
    # Define the file paths for 100% data only
    file_paths = [        
        "../Test_100/models/RF/Results/testing_numeralia.txt",
        "../Test_100/models/XGBoost/Results/testing_numeralia.txt",
        "../Test_100/models/PL-fMDP/Results/testing_numeralia_lookup_table.txt",
        "../Test_100/models/CART/Results/testing_numeralia.txt",
        "../Test_100/models/NB/Results/testing_numeralia.txt",
        "../Test_100/models/MLP/Results/testing_numeralia.txt",
        "../Test_100/models/LR/Results/testing_numeralia.txt",
    ]
    
    # Extract model names and collect data
    data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist - {file_path}")
            continue
            
        path = Path(file_path)
        model_name = path.parent.parent.name
        
        # Fix model name formatting
        if model_name == "PL-fMDP":
            model_name = "PL-fMDPs"
        elif model_name == "RF":
            model_name = "RFs"
        
        driver_metrics = parse_testing_file(file_path)
        if driver_metrics:
            for driver in range(1, 5):
                metrics = driver_metrics[driver]
                metrics['model'] = model_name
                metrics['driver'] = driver
                data.append(metrics)
                
                # Print the parsed data for verification
                print(f"\nParsed {file_path} - Driver {driver}:")
                print(f"  Model: {model_name}")
                print(f"  Precision: {format_metric(metrics.get('precision_avg'), metrics.get('precision_std'))}")
                print(f"  Recall: {format_metric(metrics.get('recall_avg'), metrics.get('recall_std'))}")
                print(f"  F1-score: {format_metric(metrics.get('f1_avg'), metrics.get('f1_std'))}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Define the desired model order
    model_order = ['PL-fMDPs', 'LR', 'CART', 'NB', 'RFs', 'XGBoost', 'MLP']
    
    # Generate a single LaTeX table with clear driver separations
    latex_output = []
    latex_output.append("\\begin{table}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Model Performance Metrics for Each Driver (100\\% Training Data)}")
    latex_output.append("\\label{tab:driver_metrics}")
    latex_output.append("\\begin{tabular}{|l|ccc|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Model} & \\textbf{Precision (Avg±std)} & \\textbf{Recall (Avg±std)} & \\textbf{F1-score (Avg±std)} \\\\")
    latex_output.append("\\hline")
    
    for driver in range(1, 5):
        latex_output.append("\\vspace{2mm}")
        latex_output.append(f"\\multicolumn{{4}}{{|c|}}{{\\textbf{{Driver {driver}}}}} \\\\")
        latex_output.append("\\hline")
        for model in model_order:
            driver_data = df[(df['model'] == model) & (df['driver'] == driver)]
            if len(driver_data) > 0:
                row = f"{model} & {format_metric(driver_data['precision_avg'].iloc[0], driver_data['precision_std'].iloc[0])} & {format_metric(driver_data['recall_avg'].iloc[0], driver_data['recall_std'].iloc[0])} & {format_metric(driver_data['f1_avg'].iloc[0], driver_data['f1_std'].iloc[0])} \\\\"
            else:
                row = f"{model} & N/A & N/A & N/A \\\\"
            latex_output.append(row)
        if driver < 4:  # Add midrule between drivers, but not after the last driver
            latex_output.append("\\midrule")
    
    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Print LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE OUTPUT FOR ALL DRIVERS:")
    print("="*80)
    for line in latex_output:
        print(line)
    
    # Save LaTeX table to file
    with open('driver_metrics_table.tex', 'w') as f:
        f.write("\n".join(latex_output))
    print("\nLaTeX table saved to 'driver_metrics_table.tex'")

if __name__ == "__main__":
    main()
