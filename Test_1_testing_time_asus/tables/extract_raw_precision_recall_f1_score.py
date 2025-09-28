#!/usr/bin/env python3
import re
import os
import ast
from pathlib import Path

def extract_raw_metrics(file_path):
    """Extract raw precision, recall, and F1 score lists from testing file"""
    metrics = {'precisions': None, 'recalls': None, 'f1_scores': None}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Patterns to find the raw lists
        patterns = {
            'precisions': r'Precisions:\s*(\[.*?\])',
            'recalls': r'Recalls:\s*(\[.*?\])', 
            'f1_scores': r'F1_scores:\s*(\[.*?\])|F1-scores:\s*(\[.*?\])'
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                # Extract the list string (handle multiple capture groups for F1)
                list_str = match.group(1) if match.group(1) else match.group(2)
                try:
                    # Safely convert string to list
                    metrics[metric_name] = ast.literal_eval(list_str)
                except (ValueError, SyntaxError):
                    print(f"Warning: Could not parse {metric_name} list in {file_path}")
                    metrics[metric_name] = None
                    
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    return metrics

def main():
    # Define the file paths
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
        
        # Train_100 files
        "../Train_100/models/RF/Results/testing_numeralia.txt",
        "../Train_100/models/XGBoost/Results/testing_numeralia.txt",
        "../Train_100/models/PL-fMDP/Results/testing_numeralia.txt",
        "../Train_100/models/CART/Results/testing_numeralia.txt",
        "../Train_100/models/NB/Results/testing_numeralia.txt",
        "../Train_100/models/MLP/Results/testing_numeralia.txt",
        "../Train_100/models/LR/Results/testing_numeralia.txt",
    ]
    
    # Organize data by percentage and model
    percentage_data = {'01': {}, '50': {}, '100': {}}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist - {file_path}")
            continue
            
        # Extract model and percentage from file path
        path = Path(file_path)
        model_name = path.parent.parent.name
        training_percent = path.parent.parent.parent.parent.name.replace('Train_', '')
        
        # Fix model name formatting
        if model_name == "PL-fMDP":
            model_name = "PL-fMDPs"
        elif model_name == "RF":
            model_name = "RFs"
        
        # Extract raw metrics
        metrics = extract_raw_metrics(file_path)
        if metrics and all(metrics.values()):  # Only if all metrics were found
            if model_name not in percentage_data[training_percent]:
                percentage_data[training_percent][model_name] = metrics
            else:
                print(f"Warning: Duplicate model {model_name} for percentage {training_percent}")
            
            print(f"Successfully extracted data from {file_path}")
        else:
            print(f"Could not extract complete metrics from {file_path}")
    
    # Create output files for each percentage
    for percentage, models_data in percentage_data.items():
        output_filename = f"raw_metrics_{percentage}percent.txt"
        
        with open(output_filename, 'w') as f:
            f.write(f"# Raw metrics for {percentage}% training data\n")
            f.write("#" + "="*50 + "\n\n")
            
            for model_name, metrics in models_data.items():
                f.write(f"{model_name}:\n")
                f.write(f"Precisions: {metrics['precisions']}\n")
                f.write(f"Recalls: {metrics['recalls']}\n")
                f.write(f"F1_scores: {metrics['f1_scores']}\n")
                f.write("\n")
        
        print(f"Created output file: {output_filename}")
    
    # Also create a combined file with all percentages
    with open("raw_metrics_all_percentages.txt", 'w') as f:
        for percentage, models_data in percentage_data.items():
            f.write(f"# {percentage}% Training Data\n")
            f.write("#" + "="*30 + "\n\n")
            
            for model_name, metrics in models_data.items():
                f.write(f"{model_name}:\n")
                f.write(f"Precisions: {metrics['precisions']}\n")
                f.write(f"Recalls: {metrics['recalls']}\n")
                f.write(f"F1_scores: {metrics['f1_scores']}\n")
                f.write("\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print("Created combined output file: raw_metrics_all_percentages.txt")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY:")
    print("="*60)
    for percentage, models_data in percentage_data.items():
        print(f"{percentage}% data: {len(models_data)} models extracted")
        for model_name in models_data.keys():
            print(f"  - {model_name}")

if __name__ == "__main__":
    main()
