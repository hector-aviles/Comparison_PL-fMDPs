#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process sample space files and analyze driver behavior')
    parser.add_argument('complete_sample_space', help='Path to complete_sample_space.csv')
    parser.add_argument('count_sample_space_auto', help='Path to count_sample_space_auto.csv')
    parser.add_argument('D_humans', help='Path to D_humans.csv')
    args = parser.parse_args()
    
    # Read the files
    print("Reading files...")
    complete_sample_space = pd.read_csv(args.complete_sample_space)
    count_sample_space_auto = pd.read_csv(args.count_sample_space_auto)
    D_humans = pd.read_csv(args.D_humans)
    
    # Task 1: Count frequencies for each driver
    print("Counting frequencies for each driver...")
    
    # Get all unique drivers
    drivers = D_humans['driver'].unique()
    
    # Create a list to store results
    all_results = []
    
    for driver in drivers:
        # Filter data for this driver
        driver_data = D_humans[D_humans['driver'] == driver]
        
        # Count occurrences of each unique row (state-action combination)
        driver_counts = driver_data.groupby(list(complete_sample_space.columns)).size().reset_index(name='count')
        
        # Merge with complete sample space to ensure we have all 512 rows
        merged = pd.merge(complete_sample_space, driver_counts, 
                         on=list(complete_sample_space.columns), 
                         how='left').fillna(0)
        
        # Add driver column and index
        merged['driver'] = driver
        merged['index'] = range(1, len(merged) + 1)
        
        all_results.append(merged)
    
    # Combine all results
    count_sample_space_by_humans = pd.concat(all_results, ignore_index=True)
    
    # Save to file
    count_sample_space_by_humans.to_csv('count_sample_space_by_humans.csv', index=False)
    print("Saved count_sample_space_by_humans.csv")
    
    # Task 2: Create normalized histograms and compute Jensen-Shannon divergence
    print("Creating histograms and computing similarities...")
    
    # Prepare auto histogram data (normalized)
    auto_hist = count_sample_space_auto['count'].values
    auto_hist_normalized = auto_hist / auto_hist.sum()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define action regions (approximate indices)
    action_regions = {
        'change_to_left': (1, 128),
        'change_to_right': (129, 256),
        'cruise': (257, 384),
        'keep': (385, 512)
    }
    
    # Calculate JS divergence for each driver
    js_results = []
    
    for driver in drivers:
        # Get this driver's data
        driver_data = count_sample_space_by_humans[count_sample_space_by_humans['driver'] == driver]
        driver_hist = driver_data['count'].values
        driver_hist_normalized = driver_hist / driver_hist.sum()
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(auto_hist_normalized, driver_hist_normalized)
        js_results.append((driver, js_div))
        
        # Plot normalized cumulative frequency
        plt.plot(driver_data['index'], driver_hist_normalized.cumsum(), 
                label=f'Driver {driver} (JS={js_div:.4f})', alpha=0.7, linewidth=1.5)
    
    # Plot auto histogram
    plt.plot(range(1, 513), auto_hist_normalized.cumsum(), 
            label='Autonomous driving', color='black', linewidth=3, linestyle='--')
    
    # Customize plot
    plt.xlabel('State-action examples', fontsize=18)
    plt.ylabel('Normalized cumulative frequency', fontsize=18)
    plt.title('Normalized cumulative frequency distribution of driving decisions', fontsize=20)
    
    # Set x-axis ticks and labels
    xticks = [1, 128, 256, 384, 512]
    plt.xticks(xticks, fontsize=10)
    
    # Add vertical lines and labels for action regions
    colors = {'change_to_left': 'red', 'change_to_right': 'blue', 'cruise': 'green', 'keep': 'purple'}
    for action, (start, end) in action_regions.items():
        plt.axvline(x=start, color=colors[action], linestyle=':', alpha=0.7)
        plt.axvline(x=end, color=colors[action], linestyle=':', alpha=0.7)
        
        # Add action labels (centered in each region)
        midpoint = (start + end) / 2
        plt.text(midpoint, plt.ylim()[1] * 0.95, action, 
                ha='center', va='top', fontsize=15, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[action], alpha=0.2))
    
    # Move legend to the left side, vertically centered, inside the plot
    plt.legend(loc='center left', fontsize=15)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Use default tight layout to adjust for legend inside the plot
    
    # Save the plot
    plt.savefig('driver_histograms_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('driver_histograms_comparison.pdf', bbox_inches='tight')
    print("Saved driver_histograms_comparison.png and .pdf")
    
    # Display Jensen-Shannon divergence results
    print("\nJensen-Shannon Divergence Results:")
    print("=" * 50)
    for driver, js_div in js_results:
        print(f"Driver {driver}: {js_div:.6f}")
    
    # Save JS results to file
    js_df = pd.DataFrame(js_results, columns=['driver', 'js_divergence'])
    js_df.to_csv('jensen_shannon_results.csv', index=False)
    print("\nSaved jensen_shannon_results.csv")

if __name__ == "__main__":
    main()
