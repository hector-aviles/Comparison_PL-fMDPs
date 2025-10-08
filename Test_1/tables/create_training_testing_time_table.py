#!/usr/bin/env python3
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path

def parse_training_file(file_path, model_name):
    """Parse training_numeralia.txt file and extract training time metrics"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # For PL-fMDP files, we need to handle both training time and AP calculation time
        if model_name == "PL-fMDP":
            # Extract training time
            training_time_avg_match = re.search(r'Avg\. training time:\s*([0-9.]+)', content)
            training_time_std_match = re.search(r'Stdev\. training time:\s*([0-9.]+)', content)
            
            if training_time_avg_match and training_time_std_match:
                metrics['training_time_avg'] = float(training_time_avg_match.group(1))
                metrics['training_time_std'] = float(training_time_std_match.group(1))
            
            # Extract all AP calculation times
            ap_times = []
            ap_avg_matches = re.findall(r'Avg\. AP calculation time:\s*([0-9.]+)', content)
            for match in ap_avg_matches:
                ap_times.append(float(match))
            
            if ap_times:
                metrics['ap_time_avg'] = np.mean(ap_times)
                metrics['ap_time_std'] = np.std(ap_times)
                
        else:
            # For other models, just extract training time
            training_time_avg_match = re.search(r'Avg\. training time:\s*([0-9.]+)', content)
            training_time_std_match = re.search(r'Stdev\. training time:\s*([0-9.]+)', content)
            
            if training_time_avg_match and training_time_std_match:
                metrics['training_time_avg'] = float(training_time_avg_match.group(1))
                metrics['training_time_std'] = float(training_time_std_match.group(1))
                
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    return metrics

def get_testing_file_path(file_path, model_name):
    """Get the correct testing file path, using lookup table for PL-fMDP if available"""
    if model_name == "PL-fMDP":
        # Try lookup table file first
        lookup_table_path = file_path.replace("testing_numeralia.txt", "testing_numeralia_lookup_table.txt")
        if os.path.exists(lookup_table_path):
            print(f"Using lookup table file: {lookup_table_path}")
            return lookup_table_path
        else:
            print(f"Lookup table file not found, using default: {file_path}")
            return file_path
    else:
        return file_path

def parse_testing_file(file_path, model_name):
    """Parse testing_numeralia.txt or testing_numeralia_lookup_table.txt file and extract testing time metrics"""
    metrics = {}
    try:
        # Get the correct file path
        actual_file_path = get_testing_file_path(file_path, model_name)
        
        if not os.path.exists(actual_file_path):
            print(f"Error: Testing file does not exist - {actual_file_path}")
            return None
            
        with open(actual_file_path, 'r') as f:
            content = f.read()
            
        # Extract testing time
        testing_time_avg_match = re.search(r'Avg\. testing time:\s*([0-9.]+)', content)
        testing_time_std_match = re.search(r'Stdev\. testing time:\s*([0-9.]+)', content)
        
        if testing_time_avg_match and testing_time_std_match:
            metrics['testing_time_avg'] = float(testing_time_avg_match.group(1))
            metrics['testing_time_std'] = float(testing_time_std_match.group(1))
        else:
            print(f"Warning: Could not extract testing time metrics from {actual_file_path}")
                
    except FileNotFoundError:
        print(f"Error: File not found - {actual_file_path}")
        return None
    except Exception as e:
        print(f"Error reading {actual_file_path}: {e}")
        return None
        
    return metrics

def format_value(avg, std, decimals=4):
    """Format average and standard deviation values"""
    if avg is None or std is None:
        return "N/A", "N/A"
    return f"{avg:.{decimals}f}", f"{std:.{decimals}f}"

def main():
    # Define file paths and model information
    training_files = []
    testing_files = []
    
    percentages = ['01', '50', '100']
    models = ['RF', 'XGBoost', 'PL-fMDP', 'CART', 'NB', 'MLP', 'LR']
    
    for percentage in percentages:
        for model in models:
            training_files.append({
                'path': f"../Train_{percentage}/models/{model}/training_numeralia.txt",
                'model': model,
                'percentage': percentage
            })
            testing_files.append({
                'path': f"../Train_{percentage}/models/{model}/Results/testing_numeralia.txt",
                'model': model,
                'percentage': percentage
            })
    
    # Parse training files
    training_data = []
    for file_info in training_files:
        if not os.path.exists(file_info['path']):
            print(f"Warning: Training file does not exist - {file_info['path']}")
            continue
            
        metrics = parse_training_file(file_info['path'], file_info['model'])
        if metrics:
            metrics.update(file_info)
            training_data.append(metrics)
            print(f"Parsed training: {file_info['path']}")
    
    # Parse testing files - REMOVE the file existence check here since we handle it in parse_testing_file
    testing_data = []
    for file_info in testing_files:
        metrics = parse_testing_file(file_info['path'], file_info['model'])
        if metrics:
            metrics.update(file_info)
            testing_data.append(metrics)
            print(f"Parsed testing: {file_info['path']}")
        else:
            print(f"Failed to parse testing file for {file_info['model']} at {file_info['percentage']}%")
    
    # Create DataFrames
    training_df = pd.DataFrame(training_data)
    testing_df = pd.DataFrame(testing_data)
    
    # Define model order and display names
    model_order = ['PL-fMDPs', 'APs', 'CART', 'MLP', 'LR', 'NB', 'RFs', 'XGBoost']
    model_mapping = {
        'PL-fMDP': 'PL-fMDPs',
        'RF': 'RFs',
        'CART': 'CART',
        'MLP': 'MLP',
        'LR': 'LR',
        'NB': 'NB',
        'XGBoost': 'XGBoost'
    }
    
    # Generate Training Time Table
    print("\n" + "="*80)
    print("TRAINING TIME TABLE:")
    print("="*80)
    
    training_latex = []
    training_latex.append("\\begin{table}[htbp!]")
    training_latex.append("\\caption{Mean and standard deviation of training time for each model instance relative to the data percentage used in Test 1 (PL-fMDPs refers to the creation of PL-fMDPs programs, and APs denotes the calculation of their action policies).}")
    training_latex.append("\\label{tab:training_exp1}")
    training_latex.append("\\begin{tabular}{lcccccc}")
    training_latex.append("\\toprule")
    training_latex.append("& \\multicolumn{6}{c}{Training time statistics (in s) per data sample} \\\\")
    training_latex.append("\\cmidrule(lr){2-7}")
    training_latex.append("& \\multicolumn{2}{c}{1\\%} & \\multicolumn{2}{c}{50\\%} & \\multicolumn{2}{c}{100\\%} \\\\")
    training_latex.append("\\cmidrule(lr){2-7}")
    training_latex.append("Model & Mean & SD & Mean & SD & Mean & SD \\\\")
    training_latex.append("\\midrule")
    
    for model_display in model_order:
        if model_display == 'APs':
            # Handle APs separately (only from PL-fMDP files)
            row_data = [model_display]
            for percentage in percentages:
                ap_data = training_df[(training_df['model'] == 'PL-fMDP') & (training_df['percentage'] == percentage)]
                if len(ap_data) > 0 and 'ap_time_avg' in ap_data.iloc[0] and 'ap_time_std' in ap_data.iloc[0]:
                    avg, std = format_value(ap_data['ap_time_avg'].iloc[0], ap_data['ap_time_std'].iloc[0], 2)
                    row_data.extend([avg, std])
                else:
                    row_data.extend(["N/A", "N/A"])
            training_latex.append(" & ".join(row_data) + " \\\\")
            
        elif model_display == 'PL-fMDPs':
            # Handle PL-fMDPs training time
            row_data = [model_display]
            for percentage in percentages:
                pl_data = training_df[(training_df['model'] == 'PL-fMDP') & (training_df['percentage'] == percentage)]
                if len(pl_data) > 0 and 'training_time_avg' in pl_data.iloc[0] and 'training_time_std' in pl_data.iloc[0]:
                    avg, std = format_value(pl_data['training_time_avg'].iloc[0], pl_data['training_time_std'].iloc[0], 2)
                    row_data.extend([avg, std])
                else:
                    row_data.extend(["N/A", "N/A"])
            training_latex.append(" & ".join(row_data) + " \\\\")
            
        else:
            # Handle other models
            row_data = [model_display]
            original_model = next((k for k, v in model_mapping.items() if v == model_display), model_display)
            for percentage in percentages:
                model_data = training_df[(training_df['model'] == original_model) & (training_df['percentage'] == percentage)]
                if len(model_data) > 0 and 'training_time_avg' in model_data.iloc[0] and 'training_time_std' in model_data.iloc[0]:
                    avg, std = format_value(model_data['training_time_avg'].iloc[0], model_data['training_time_std'].iloc[0], 4)
                    row_data.extend([avg, std])
                else:
                    row_data.extend(["N/A", "N/A"])
            training_latex.append(" & ".join(row_data) + " \\\\")
    
    training_latex.append("\\bottomrule")
    training_latex.append("\\end{tabular}")
    training_latex.append("\\end{table}")
    
    # Generate Testing Time Table
    print("\n" + "="*80)
    print("TESTING TIME TABLE:")
    print("="*80)
    
    testing_latex = []
    testing_latex.append("\\begin{table}[htbp!]")
    testing_latex.append("\\caption{Mean and standard deviation of testing time for each model and set of data in Test 1.}")
    testing_latex.append("\\label{tab:testing_exp1}")
    testing_latex.append("\\begin{tabular}{lcccccc}")
    testing_latex.append("\\toprule")
    testing_latex.append("& \\multicolumn{6}{c}{Testing time statistics (in s) per data sample} \\\\")
    testing_latex.append("\\cmidrule(lr){2-7}")
    testing_latex.append("& \\multicolumn{2}{c}{1\\%} & \\multicolumn{2}{c}{50\\%} & \\multicolumn{2}{c}{100\\%} \\\\")
    testing_latex.append("\\cmidrule(lr){2-7}")
    testing_latex.append("Model & Mean & SD & Mean & SD & Mean & SD \\\\")
    testing_latex.append("\\midrule")
    
    # For testing table, we need to handle APs separately (they come from PL-fMDP training files)
    for model_display in ['APs'] + [m for m in model_order if m not in ['PL-fMDPs', 'APs']]:
        if model_display == 'APs':
            # Handle APs testing time (from PL-fMDP training files)
            row_data = [model_display]
            for percentage in percentages:
                ap_data = training_df[(training_df['model'] == 'PL-fMDP') & (training_df['percentage'] == percentage)]
                if len(ap_data) > 0 and 'ap_time_avg' in ap_data.iloc[0] and 'ap_time_std' in ap_data.iloc[0]:
                    avg, std = format_value(ap_data['ap_time_avg'].iloc[0], ap_data['ap_time_std'].iloc[0], 2)
                    row_data.extend([avg, std])
                else:
                    row_data.extend(["N/A", "N/A"])
            testing_latex.append(" & ".join(row_data) + " \\\\")
            
        else:
            # Handle other models' testing time
            row_data = [model_display]
            original_model = next((k for k, v in model_mapping.items() if v == model_display), model_display)
            for percentage in percentages:
                model_data = testing_df[(testing_df['model'] == original_model) & (testing_df['percentage'] == percentage)]
                if len(model_data) > 0 and 'testing_time_avg' in model_data.iloc[0] and 'testing_time_std' in model_data.iloc[0]:
                    avg, std = format_value(model_data['testing_time_avg'].iloc[0], model_data['testing_time_std'].iloc[0], 4)
                    row_data.extend([avg, std])
                else:
                    row_data.extend(["N/A", "N/A"])
            testing_latex.append(" & ".join(row_data) + " \\\\")
    
    testing_latex.append("\\bottomrule")
    testing_latex.append("\\end{tabular}")
    testing_latex.append("\\end{table}")
    
    # Save tables to files
    with open('training_time_table.tex', 'w') as f:
        f.write("\n".join(training_latex))
    print("Training time table saved to 'training_time_table.tex'")
    
    with open('testing_time_table.tex', 'w') as f:
        f.write("\n".join(testing_latex))
    print("Testing time table saved to 'testing_time_table.tex'")
    
    # Print tables for verification
    print("\n" + "="*80)
    print("TRAINING TIME TABLE (LATEX):")
    print("="*80)
    for line in training_latex:
        print(line)
    
    print("\n" + "="*80)
    print("TESTING TIME TABLE (LATEX):")
    print("="*80)
    for line in testing_latex:
        print(line)

if __name__ == "__main__":
    main()
