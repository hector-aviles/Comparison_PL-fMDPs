import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

# Define the file paths - ONLY 100% DATA
file_paths = [
    # Test_100 files only
    "../Test_100/models/RF/Results/testing_numeralia.txt",
    "../Test_100/models/XGBoost/Results/testing_numeralia.txt",
    "../Test_100/models/PL-fMDP/Results/testing_numeralia.txt",
    "../Test_100/models/CART/Results/testing_numeralia.txt",
    "../Test_100/models/NB/Results/testing_numeralia.txt",
    "../Test_100/models/MLP/Results/testing_numeralia.txt",
    "../Test_100/models/LR/Results/testing_numeralia.txt"
]

# Define models and percentages (only 100% now)
models = ["PL-fMDP", "LR", "CART", "NB", "RF", "XGBoost", "MLP"]
percentages = ["100"]  # Only 100% percentage
drivers = [1, 2, 3, 4]

# Initialize data structures to store results
mean_f1_scores = {driver: {model: {perc: None for perc in percentages} for model in models} for driver in drivers}
std_f1_scores = {driver: {model: {perc: None for perc in percentages} for model in models} for driver in drivers}

def parse_regular_file(content, driver, model, perc):
    """Parse regular files (non-PL-fMDP) with multiple possible formats"""
    # Try different patterns for driver section
    driver_patterns = [
        f"Metrics for Driver {driver}:",
        f"Driver {driver} Metrics:",
        f"Driver {driver}:"
    ]
    
    # Try different patterns for F1 score
    f1_patterns = [
        r"Average f1: ([\d.]+), Std Dev f1: ([\d.]+)",
        r"F1: ([\d.]+), Std Dev: ([\d.]+)",
        r"Average F1: ([\d.]+), Std Dev F1: ([\d.]+)",
        r"F1 score: ([\d.]+), Std Dev: ([\d.]+)"
    ]
    
    lines = content.split('\n')
    in_driver_section = False
    
    for line in lines:
        # Check if we're entering a driver section
        if not in_driver_section:
            for pattern in driver_patterns:
                if pattern in line:
                    in_driver_section = True
                    break
            continue
        
        # If we're in a driver section, look for F1 score
        if in_driver_section:
            for pattern in f1_patterns:
                match = re.search(pattern, line)
                if match:
                    mean_f1 = float(match.group(1))
                    std_f1 = float(match.group(2))
                    mean_f1_scores[driver][model][perc] = mean_f1
                    std_f1_scores[driver][model][perc] = std_f1
                    return True
            
            # Check if we're entering a new driver section (end of current section)
            if line.strip() and any(f"Driver {d}" in line for d in drivers if d != driver):
                return False
    
    return False

def parse_pl_fmdp_file(content, driver, model, perc):
    """Parse PL-fMDP files"""
    driver_pattern = f"Performance for Driver {driver}:"
    f1_pattern = r"Average F1_scores: ([\d.]+)"
    std_pattern = r"Std Dev F1_scores: ([\d.]+)"
    
    lines = content.split('\n')
    in_driver_section = False
    found_f1 = False
    
    for line in lines:
        if driver_pattern in line:
            in_driver_section = True
            continue
        
        if in_driver_section:
            if not found_f1:
                match = re.search(f1_pattern, line)
                if match:
                    mean_f1 = float(match.group(1))
                    mean_f1_scores[driver][model][perc] = mean_f1
                    found_f1 = True
                    continue
            
            if found_f1:
                match = re.search(std_pattern, line)
                if match:
                    std_f1 = float(match.group(1))
                    std_f1_scores[driver][model][perc] = std_f1
                    return True
            
            if line.strip() and "Performance for Driver" in line:
                # Found next driver section
                return False
    
    return False

# Read and parse all files
print("Reading and parsing files...")
for file_path in file_paths:
    # Extract model and percentage from file path
    path_parts = file_path.split('/')
    perc = path_parts[1].replace('Test_', '')
    model = path_parts[3]
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        continue
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    print(f"\nProcessing: {file_path}")
    
    # Parse for each driver
    for driver in drivers:
        if model == "PL-fMDP":
            success = parse_pl_fmdp_file(content, driver, model, perc)
        else:
            success = parse_regular_file(content, driver, model, perc)
        
        if success:
            print(f"  Driver {driver}: F1 = {mean_f1_scores[driver][model][perc]:.4f} ± {std_f1_scores[driver][model][perc]:.4f}")
        else:
            print(f"  Driver {driver}: Data not found")

# Print the extracted data for verification
print("\n" + "="*80)
print("EXTRACTED DATA VERIFICATION - 100% DATA ONLY")
print("="*80)

for driver in drivers:
    print(f"\nDriver {driver}:")
    print("-" * 50)
    for model in models:
        mean_val = mean_f1_scores[driver][model]["100"]
        std_val = std_f1_scores[driver][model]["100"]
        if mean_val is not None and std_val is not None:
            print(f"{model:10}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{model:10}: N/A")

# Create LaTeX table for 100% data only
print("\n" + "="*80)
print("LATEX TABLE FOR 100% DATA")
print("="*80)

# Create LaTeX table
latex_table = """\\begin{table}[htbp!]
\\centering
\\caption{Mean and standard deviation of F1 scores for each model and driver using 100\% of training data.}
{\\tablefont\\begin{tabular}{@{\\extracolsep{0.3cm}}lcccc}
\\topline
& \\multicolumn{4}{c}{Driver} \\\\
\\cmidrule(lr){2-5}
Model & 1 & 2 & 3 & 4 \\\\
\\midline
"""

# Add data rows
for model in models:
    row = f"{model:10}"
    for driver in drivers:
        mean_val = mean_f1_scores[driver][model]["100"]
        std_val = std_f1_scores[driver][model]["100"]
        if mean_val is not None and std_val is not None:
            row += f" & {mean_val:.3f} $\\pm$ {std_val:.3f}"
        else:
            row += " & N/A"
    row += " \\\\\n"
    latex_table += row

# Close the table
latex_table += """\\botline
\\end{tabular}}
\\label{tab:f1_scores_100percent}
\\end{table}"""

print(latex_table)

# Save LaTeX table to file
with open('f1_scores_100percent.tex', 'w') as f:
    f.write(latex_table)

print("\nLaTeX table saved to 'f1_scores_100percent.tex'")

# Create a single comprehensive plot for 100% data only
print("\n" + "="*80)
print("CREATING COMPREHENSIVE PLOT FOR 100% DATA")
print("="*80)

# Define colors with high saturation for better contrast
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628']
model_labels = ["PL-fMDPs", "LR", "CART", "NB", "RFs", "XGBoost", "MLPs"]

# Create a single figure with subplots for each driver for 100% data
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('F1 Score Comparison Across Models - 100% Data', fontsize=20, fontweight='bold')

for i, driver in enumerate(drivers):
    ax = axes[i//2, i%2]
    
    # Prepare data for plotting - only for 100% percentage
    x_pos = np.arange(len(models))
    width = 0.7
    
    # Plot each model for 100% percentage
    means = []
    stds = []
    
    for j, model in enumerate(models):
        mean_val = mean_f1_scores[driver][model]["100"]
        std_val = std_f1_scores[driver][model]["100"]
        
        if mean_val is not None and std_val is not None:
            means.append(mean_val)
            stds.append(std_val)
        else:
            means.append(0)
            stds.append(0)
    
    # Plot bars with error bars
    bars = ax.bar(x_pos, means, width, yerr=stds, 
                 color=colors, alpha=0.8, ecolor='black', capsize=5)
    
    # Add value labels on top of bars with both mean and stddev
    for k, (mean_val, std_val) in enumerate(zip(means, stds)):
        if mean_val > 0:
            label_text = f'{mean_val:.3f} ± {std_val:.3f}'
            ax.text(x_pos[k], mean_val + std_val + 0.03, label_text, 
                   ha='center', va='bottom', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Customize the subplot
    ax.set_title(f'Driver {driver}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add a nice frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Save the plot
plt.savefig('driver_comparison_100_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('driver_comparison_100_plot.pdf', bbox_inches='tight')
plt.close()

print("Plot for 100% data saved as 'driver_comparison_100_plot.png' and 'driver_comparison_100_plot.pdf'")

print("\nAll outputs generated successfully!")
