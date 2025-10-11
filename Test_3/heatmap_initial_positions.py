##########################################################################
# Python3
#####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from matplotlib import rc
import matplotlib as mpl
import matplotlib.colors as mcolors

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "font.family": "serif"
})

# Read the CSV file
print("Reading CSV file...")
data = pd.read_csv('experiments_Inteligencia_Artificial.csv', usecols=['iteration', "av_pos.x","av_pos.y",'car1_pos.x', 'car1_pos.y', 'car2_pos.x', 'car2_pos.y','car3_pos.x', 'car3_pos.y', 'car4_pos.x', 'car4_pos.y'], low_memory=False)

print(f"Total rows in CSV: {len(data)}")
print(f"Columns: {data.columns.tolist()}")

condition = (data['iteration'] == 1)
filtered_data = data[condition]
print(f"Rows with iteration == 10: {len(filtered_data)}")

# Debug: Check if we have any data at all
if len(filtered_data) == 0:
    print("WARNING: No data found for iteration == 10!")
else:
    print("Sample of filtered data:")
    print(filtered_data[['av_pos.x', 'av_pos.y', 'car1_pos.x', 'car1_pos.y']].head())

# Separate data for different vehicle types
av_x = filtered_data['av_pos.x'].tolist()
av_y = filtered_data['av_pos.y'].tolist()

# Other vehicles (all cars except AV)
other_x = (filtered_data['car1_pos.x'].tolist() + filtered_data['car2_pos.x'].tolist() + 
           filtered_data['car3_pos.x'].tolist() + filtered_data['car4_pos.x'].tolist())

other_y = (filtered_data['car1_pos.y'].tolist() + filtered_data['car2_pos.y'].tolist() + 
           filtered_data['car3_pos.y'].tolist() + filtered_data['car4_pos.y'].tolist())

print(f"\nRaw data counts:")
print(f"AV positions: {len(av_x)}")
print(f"Other vehicles positions: {len(other_x)}")

# Convert to arrays
av_x = np.array(av_x)
av_y = np.array(av_y)
other_x = np.array(other_x)
other_y = np.array(other_y)

print(f"\nBefore removing zeros:")
print(f"AV - any zeros in x: {np.any(av_x == 0)}, any zeros in y: {np.any(av_y == 0)}")
print(f"Other - any zeros in x: {np.any(other_x == 0)}, any zeros in y: {np.any(other_y == 0)}")
print(f"AV x range: {np.min(av_x):.2f} - {np.max(av_x):.2f}")
print(f"AV y range: {np.min(av_y):.2f} - {np.max(av_y):.2f}")
print(f"Other x range: {np.min(other_x):.2f} - {np.max(other_x):.2f}")
print(f"Other y range: {np.min(other_y):.2f} - {np.max(other_y):.2f}")

# Remove zeros for each dataset separately
av_mask = ~np.logical_or(av_x == 0, av_y == 0)
other_mask = ~np.logical_or(other_x == 0, other_y == 0)

print(f"\nZero removal:")
print(f"AV - removing {np.sum(~av_mask)} zero positions")
print(f"Other - removing {np.sum(~other_mask)} zero positions")

av_x = av_x[av_mask]
av_y = av_y[av_mask]
other_x = other_x[other_mask]
other_y = other_y[other_mask]

print(f"\nAfter removing zeros:")
print(f"AV positions remaining: {len(av_x)}")
print(f"Other vehicles positions remaining: {len(other_x)}")
print(f"AV x range: {np.min(av_x):.2f} - {np.max(av_x):.2f}")
print(f"AV y range: {np.min(av_y):.2f} - {np.max(av_y):.2f}")
print(f"Other x range: {np.min(other_x):.2f} - {np.max(other_x):.2f}")
print(f"Other y range: {np.min(other_y):.2f} - {np.max(other_y):.2f}")

# Check if we have any valid data left
if len(av_x) == 0:
    print("WARNING: No valid AV data after zero removal!")
if len(other_x) == 0:
    print("WARNING: No valid other vehicles data after zero removal!")

def myplot(x, y, s, bins=500):
   heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 130], [-3.5, 3.5]])
   heatmap = gaussian_filter(heatmap, sigma=s)
   extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
   return heatmap.T, extent

# Create individual heatmaps for each vehicle type
print(f"\nCreating heatmaps...")
img_other, extent = myplot(other_x, other_y, s=2)
img_av, extent = myplot(av_x, av_y, s=5)

print(f"\nHeatmap statistics:")
print(f"AV frequency range: {np.min(img_av):.4f} - {np.max(img_av):.4f}")
print(f"Other vehicles frequency range: {np.min(img_other):.4f} - {np.max(img_other):.4f}")
print(f"AV non-zero pixels: {np.sum(img_av > 0)}")
print(f"Other non-zero pixels: {np.sum(img_other > 0)}")

# Check if other vehicles heatmap has any significant values
if np.max(img_other) < 0.1:
    print("WARNING: Other vehicles heatmap has very low values!")
    print("This could mean:")
    print("1. Other vehicles are not present in the data")
    print("2. Other vehicles are outside the range [0,130] x [-3.5,3.5]")
    print("3. Other vehicles have mostly zero coordinates")

# Create highly saturated custom colormaps
colors_red = ['#FFE6E6', '#FF9999', '#FF4D4D', '#CC0000', '#990000', '#660000']
cmap_red = mcolors.LinearSegmentedColormap.from_list('saturated_red', colors_red, N=256)

colors_blue = ['#E6F3FF', '#99D6FF', '#4DB8FF', '#0080FF', '#0066CC', '#004C99']
cmap_blue = mcolors.LinearSegmentedColormap.from_list('saturated_blue', colors_blue, N=256)

# Use individual vmax for each dataset to maximize contrast
vmax_av = max(0.1, np.max(img_av)) * 1.2
vmax_other = max(0.1, np.max(img_other)) * 1.2

print(f"Using vmax_av: {vmax_av:.4f}, vmax_other: {vmax_other:.4f}")

# Create separate plots for each vehicle type with white background
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Set white background for the entire figure
fig.patch.set_facecolor('white')

# Plot 1: AV only with white background
im1 = ax1.imshow(img_av, extent=extent, origin='lower', cmap=cmap_red, 
                interpolation='nearest', aspect=10, vmax=vmax_av)
ax1.set_title('Self-driving car initial positions', fontsize=12)#, #fontweight='bold')
# Use dashed line for lane marking
ax1.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=12)
ax1.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=12)
ax1.set_ylim([-2.5, 2.5])
ax1.set_xlim([0, 130])
# Set white background for the plot area
ax1.set_facecolor('white')
# Add Lane Marking text
ax1.text(5, 0.25, 'Lane marking', dict(size=9), color='black',
         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))

# Plot 2: Other vehicles only with white background
im2 = ax2.imshow(img_other, extent=extent, origin='lower', cmap=cmap_blue, 
                interpolation='nearest', aspect=10, vmax=vmax_other)
ax2.set_title('Obstacle vehicles initial positions', fontsize=12)#, #fontweight='bold')
# Use dashed line for lane marking
ax2.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=12)
ax2.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=12)
ax2.set_ylim([-2.5, 2.5])
ax2.set_xlim([0, 130])
# Set white background for the plot area
ax2.set_facecolor('white')
# Add Lane Marking text
ax2.text(5, 0.25, 'Lane marking', dict(size=9), color='black',
         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))

# Add compact colorbars using the standard approach with adjusted positioning
cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.4, pad=0.05, aspect=20)
cbar1.set_label('Frequency', fontsize=9)

cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.4, pad=0.05, aspect=20)
cbar2.set_label('Frequency', fontsize=9)

plt.tight_layout()
plt.savefig("vehicle_positions_separate.pdf", bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
plt.show()

print("\nDebugging complete!")
