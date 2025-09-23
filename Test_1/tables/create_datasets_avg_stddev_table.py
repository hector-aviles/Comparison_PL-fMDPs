#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def main():
    # Define the file paths
    file_paths = [
        "../Train_01/summary.csv",
        "../Train_50/summary.csv", 
        "../Train_100/summary.csv"
    ]
    
    # Dictionary to store data for each percentage
    data_dict = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist - {file_path}")
            continue
            
        # Extract percentage from file path
        percentage = file_path.split('/')[1].replace('Train_', '').replace('%', '')
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Calculate average and standard deviation for each column
            stats = {}
            for col in ['train_rows', 'tune_rows', 'test_rows']:
                if col in df.columns:
                    stats[f'{col}_avg'] = df[col].mean()
                    stats[f'{col}_std'] = df[col].std()
                else:
                    print(f"Warning: Column '{col}' not found in {file_path}")
                    stats[f'{col}_avg'] = np.nan
                    stats[f'{col}_std'] = np.nan
            
            # Store the statistics
            data_dict[percentage] = stats
            
            print(f"Processed {file_path}:")
            print(f"  Train rows: {stats['train_rows_avg']:.0f} ± {stats['train_rows_std']:.0f}")
            print(f"  Tune rows:  {stats['tune_rows_avg']:.0f} ± {stats['tune_rows_std']:.0f}")
            print(f"  Test rows:  {stats['test_rows_avg']:.0f} ± {stats['test_rows_std']:.0f}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Define the order of percentages to display
    percentages = ['01', '50', '100']
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append("\\begin{table}[ht!]")
    latex_output.append("\\caption{Sample size and statistics of examples used for training, tuning and testing models}")
    latex_output.append("\\label{tab:size_stats}")
    latex_output.append("\\begin{tabular}{c|c|ccc}")
    latex_output.append("\\toprule")
    latex_output.append("Data & Sample & \\multicolumn{3}{c}{Dataset Statistics} \\\\")
    latex_output.append("\\cmidrule(lr){3-5}")
    latex_output.append("\\% & size & $\\mathcal{D}_{train}$ & $\\mathcal{D}_{tune}$ & $\\mathcal{D}_{test}$ \\\\")
    latex_output.append("\\midrule")
    
    for percentage in percentages:
        if percentage in data_dict:
            stats = data_dict[percentage]
            
            # Get sample size (assuming it's the same across all rows, take from first row)
            try:
                df = pd.read_csv(f"../Train_{percentage}/summary.csv")
                sample_size = int(df['train_rows'].sum() + df['tune_rows'].sum() + df['test_rows'].sum())
            except:
                sample_size = 0
            
            # Format the values
            train_str = f"{stats['train_rows_avg']:.0f} $\\pm$ {stats['train_rows_std']:.0f}" if not pd.isna(stats['train_rows_avg']) else "N/A"
            tune_str = f"{stats['tune_rows_avg']:.0f} $\\pm$ {stats['tune_rows_std']:.0f}" if not pd.isna(stats['tune_rows_avg']) else "N/A"
            test_str = f"{stats['test_rows_avg']:.0f} $\\pm$ {stats['test_rows_std']:.0f}" if not pd.isna(stats['test_rows_avg']) else "N/A"
            
            latex_output.append(f"{percentage}\\% & {sample_size:,} & {train_str} & {tune_str} & {test_str} \\\\")
        else:
            latex_output.append(f"{percentage}\\% & N/A & N/A & N/A & N/A \\\\")
    
    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")
    
    # Print LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE OUTPUT:")
    print("="*80)
    for line in latex_output:
        print(line)
    
    # Save LaTeX table to file
    with open('statistics_table_avg_stddev_training_datasets.tex', 'w') as f:
        f.write("\n".join(latex_output))
    print("\nLaTeX table saved to 'statistics_table_avg_stddev_training_datasets.tex'")
    
    # Also create a simpler text version for verification
    print("\n" + "="*80)
    print("TEXT TABLE FOR VERIFICATION:")
    print("="*80)
    print("Data % | Sample size | D_train (Avg±std) | D_tune (Avg±std) | D_test (Avg±std)")
    print("-" * 85)
    
    for percentage in percentages:
        if percentage in data_dict:
            stats = data_dict[percentage]
            try:
                df = pd.read_csv(f"../Train_{percentage}/summary.csv")
                sample_size = int(df['train_rows'].sum() + df['tune_rows'].sum() + df['test_rows'].sum())
            except:
                sample_size = 0
            
            train_str = f"{stats['train_rows_avg']:.0f} ± {stats['train_rows_std']:.0f}" if not pd.isna(stats['train_rows_avg']) else "N/A"
            tune_str = f"{stats['tune_rows_avg']:.0f} ± {stats['tune_rows_std']:.0f}" if not pd.isna(stats['tune_rows_avg']) else "N/A"
            test_str = f"{stats['test_rows_avg']:.0f} ± {stats['test_rows_std']:.0f}" if not pd.isna(stats['test_rows_avg']) else "N/A"
            
            print(f"{percentage}%     | {sample_size:>10,} | {train_str:>18} | {tune_str:>16} | {test_str:>15}")

if __name__ == "__main__":
    main()
