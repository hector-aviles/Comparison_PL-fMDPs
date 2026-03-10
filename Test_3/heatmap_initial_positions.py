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
    
# Read transverse_coords file
print("\nReading transverse_coords file...")
try:
    # First check what columns are actually in the file
    transverse_data_sample = pd.read_csv('transverse_initial_coords.csv', nrows=0)
    print(f"Actual columns in file: {transverse_data_sample.columns.tolist()}")
    
    # Read the data - pandas should handle quoted headers automatically
    transverse_data = pd.read_csv('transverse_initial_coords.csv')
    print(f"Transverse data loaded: {len(transverse_data)} rows")
    print(f"Transverse data columns: {transverse_data.columns.tolist()}")
    
    # Use the actual column names from the file
    if len(transverse_data.columns) >= 2:
        x_col = transverse_data.columns[0]
        y_col = transverse_data.columns[1]
        print(f"Using columns: '{x_col}' for x and '{y_col}' for y")
        
        print(f"Transverse data range - x: {transverse_data[x_col].min():.2f} to {transverse_data[x_col].max():.2f}")
        print(f"Transverse data range - y: {transverse_data[y_col].min():.2f} to {transverse_data[y_col].max():.2f}")
        
        # Remove zeros from transverse data
        transverse_mask = ~np.logical_or(transverse_data[x_col] == 0, transverse_data[y_col] == 0)
        transverse_x = transverse_data[x_col][transverse_mask].values
        transverse_y = transverse_data[y_col][transverse_mask].values
        print(f"Transverse data after zero removal: {len(transverse_x)} positions")
    else:
        print("WARNING: Not enough columns in transverse_initial_coords.csv")
        transverse_x = np.array([])
        transverse_y = np.array([])
    
except FileNotFoundError:
    print("WARNING: transverse_initial_coords.csv file not found!")
    transverse_x = np.array([])
    transverse_y = np.array([])
except Exception as e:
    print(f"WARNING: Error reading transverse_initial_coords.csv: {e}")
    transverse_x = np.array([])
    transverse_y = np.array([])    
    
# Read world_15_initial_coords file
print("\nReading world_15_initial_coords file...")
try:
    # First check what columns are actually in the file
    world_data_sample = pd.read_csv('world_15_initial_coords.csv', nrows=0)
    print(f"Actual columns in file: {world_data_sample.columns.tolist()}")
    
    # Read the data - pandas should handle quoted headers automatically
    world_data = pd.read_csv('world_15_initial_coords.csv')
    print(f"World 15 data loaded: {len(world_data)} rows")
    print(f"World 15 data columns: {world_data.columns.tolist()}")
    
    # Use the actual column names from the file
    if len(world_data.columns) >= 2:
        x_col = world_data.columns[0]
        y_col = world_data.columns[1]
        print(f"Using columns: '{x_col}' for x and '{y_col}' for y")
        
        print(f"World 15 data range - x: {world_data[x_col].min():.2f} to {world_data[x_col].max():.2f}")
        print(f"World 15 data range - y: {world_data[y_col].min():.2f} to {world_data[y_col].max():.2f}")
        
        # Remove zeros from world data
        world_mask = ~np.logical_or(world_data[x_col] == 0, world_data[y_col] == 0)
        world_x = world_data[x_col][world_mask].values
        world_y = world_data[y_col][world_mask].values
        print(f"World 15 data after zero removal: {len(world_x)} positions")
    else:
        print("WARNING: Not enough columns in world_15_initial_coords.csv")
        world_x = np.array([])
        world_y = np.array([])
    
except FileNotFoundError:
    print("WARNING: world_15_initial_coords.csv file not found!")
    world_x = np.array([])
    world_y = np.array([])
except Exception as e:
    print(f"WARNING: Error reading world_15_initial_coords.csv: {e}")
    world_x = np.array([])
    world_y = np.array([])
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

def myplot(x, y, s, bins=500, range_extent=None):
    if range_extent is None:
        range_extent = [[0, 130], [-3.5, 3.5]]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_extent)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

# Create individual heatmaps for each vehicle type
print(f"\nCreating heatmaps...")
img_other, extent = myplot(other_x, other_y, s=2)
img_av, extent = myplot(av_x, av_y, s=2)

# Create heatmap for world data and combine with other vehicles
if len(world_x) > 0:
    img_world, extent_world = myplot(world_x, world_y, s=2)
    print(f"World 15 data frequency range: {np.min(img_world):.4f} - {np.max(img_world):.4f}")
    
    # Combine world data with other vehicles data
    img_other_combined = img_other + img_world
    print(f"Combined other+world frequency range: {np.min(img_other_combined):.4f} - {np.max(img_other_combined):.4f}")
else:
    img_other_combined = img_other
    print("No world data found, using only other vehicles data")

# Create heatmap for transverse data with vertical orientation
if len(transverse_x) > 0:
    # Set specific x-range from 61 to 68
    img_transverse, extent_transverse = myplot(transverse_x, transverse_y, s=1,  # Reduced sigma for less smoothing
                                             range_extent=[[62.25, 69.25], [-28, -10.0]])
    print(f"Transverse plot x-range: 61 to 68")
    
    # Print actual frequency statistics for transverse data
    print(f"Transverse data actual frequency range: {np.min(img_transverse):.4f} - {np.max(img_transverse):.4f}")
    print(f"Transverse data non-zero pixels: {np.sum(img_transverse > 0)}")
    
    # If the max frequency is very low, we might need to adjust the binning
    if np.max(img_transverse) < 0.5:
        print("Note: Transverse frequencies are low, using fixed vmax=1.0 for better color distribution")
else:
    # Create empty heatmap if no transverse data
    img_transverse, extent_transverse = np.zeros((500, 500)), [62.25, 69.25, -28, -10.0]

print(f"\nHeatmap statistics:")
print(f"AV frequency range: {np.min(img_av):.4f} - {np.max(img_av):.4f}")
print(f"Other vehicles frequency range: {np.min(img_other):.4f} - {np.max(img_other):.4f}")
if len(world_x) > 0:
    print(f"World 15 frequency range: {np.min(img_world):.4f} - {np.max(img_world):.4f}")
    print(f"Combined other+world frequency range: {np.min(img_other_combined):.4f} - {np.max(img_other_combined):.4f}")
if len(transverse_x) > 0:
    print(f"Transverse frequency range: {np.min(img_transverse):.4f} - {np.max(img_transverse):.4f}")

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

colors_green = ['#E6FFE6', '#99FF99', '#4DFF4D', '#00CC00', '#009900', '#006600']
cmap_green = mcolors.LinearSegmentedColormap.from_list('saturated_green', colors_green, N=256)

# Use individual vmax for each dataset to maximize contrast
vmax_av = max(0.1, np.max(img_av)) * 1.2
vmax_other = max(0.1, np.max(img_other_combined)) * 1.2  # Fixed: removed duplicate assignment
vmax_transverse = max(0.1, np.max(img_transverse)) * 1.2 if len(transverse_x) > 0 else 1.0

print(f"Using vmax_av: {vmax_av:.4f}, vmax_other: {vmax_other:.4f}, vmax_transverse: {vmax_transverse:.4f}")

# Create three separate figures

# Figure 1: AV only
fig1, ax1 = plt.subplots(figsize=(10, 4))
im1 = ax1.imshow(img_av, extent=extent, origin='lower', cmap=cmap_red, 
                interpolation='nearest', aspect=10, vmax=vmax_av)
#ax1.set_title('Self-driving car initial positions', fontsize=12)
ax1.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=20)
ax1.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=20)
ax1.set_ylim([-3.5, 3.5])
ax1.set_xlim([0, 130])
ax1.set_facecolor('white')
ax1.text(5, 0.25, 'Lane marking', dict(size=14), color='black',
         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))
cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05, aspect=20)
cbar1.set_label('Frequency', fontsize=18)
# Set integer ticks for AV colorbar
cbar1.set_ticks(np.arange(0, int(vmax_av) + 1, 1))
fig1.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("av_positions.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
plt.show()

# Figure 2: Other vehicles only
fig2, ax2 = plt.subplots(figsize=(10, 4))
im2 = ax2.imshow(img_other_combined, extent=extent, origin='lower', cmap=cmap_blue, interpolation='nearest', aspect=10, vmax=vmax_other)
#ax2.set_title('Obstacle vehicles initial positions', fontsize=12)
ax2.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=20)
ax2.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=20)
ax2.set_ylim([-3.5, 3.5])
ax2.set_xlim([0, 130])
ax2.set_facecolor('white')
ax2.text(5, 0.25, 'Lane marking', dict(size=14), color='black',
         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))
cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05, aspect=20)
cbar2.set_label('Frequency', fontsize=18)
# Set integer ticks for Other vehicles colorbar
cbar2.set_ticks(np.arange(0, int(vmax_other) + 1, 1))
fig2.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("other_vehicles_positions.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
plt.show()

# Figure 3: Transverse data only
fig3, ax3 = plt.subplots(figsize=(6, 6))

# Use fixed vmax=1.0 for transverse plot to ensure full color range
vmax_transverse_fixed = 1.0
im3 = ax3.imshow(img_transverse, extent=extent_transverse, origin='lower', cmap=cmap_green, 
                interpolation='nearest', aspect=0.7, vmax=vmax_transverse_fixed)

#ax3.set_title('Transverse Coordinates', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=20)
ax3.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=20)
ax3.set_ylim([-28, -10.0])
ax3.set_xlim([62.25, 69.25])
ax3.set_facecolor('white')
ax3.set_xticks([62.25, 69.25])
ax3.tick_params(axis='x', which='major', labelsize=10)

# Add vertical line as Lane Marking
lane_x = 65.75
ax3.axvline(x=lane_x, color='black', linestyle='dashed', linewidth=1)

# Add Lane Marking text
ax3.text(lane_x + 0.3, -15, 'Lane marking', dict(size=14), color='black',
         bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9),
         rotation=90)

cbar3 = fig3.colorbar(im3, ax=ax3, shrink=0.8, pad=0.05, aspect=20)
cbar3.set_label('Frequency', fontsize=18)
# Set ticks from 0 to 1 with step 0.2 for better readability
cbar3.set_ticks(np.arange(0, 1.1, 0.2))  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
fig3.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("transverse_positions.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
plt.show()

print("\nDebugging complete!")
