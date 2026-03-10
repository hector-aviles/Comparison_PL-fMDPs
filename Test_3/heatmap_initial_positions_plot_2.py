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

# Read the coords.csv file
print("Reading coords.csv file...")
try:
    # First check what columns are actually in the file
    coords_sample = pd.read_csv('coords.csv', nrows=0)
    print(f"Actual columns in file: {coords_sample.columns.tolist()}")
    
    # Read the data
    coords_data = pd.read_csv('coords.csv')
    print(f"Coords data loaded: {len(coords_data)} rows")
    print(f"Coords data columns: {coords_data.columns.tolist()}")
    
    # Use the actual column names from the file
    if len(coords_data.columns) >= 2:
        x_col = coords_data.columns[0]
        y_col = coords_data.columns[1]
        print(f"Using columns: '{x_col}' for x and '{y_col}' for y")
        
        print(f"Coords data range - x: {coords_data[x_col].min():.2f} to {coords_data[x_col].max():.2f}")
        print(f"Coords data range - y: {coords_data[y_col].min():.2f} to {coords_data[y_col].max():.2f}")
        
        # Remove zeros from coords data
        coords_mask = ~np.logical_or(coords_data[x_col] == 0, coords_data[y_col] == 0)
        coords_x = coords_data[x_col][coords_mask].values
        coords_y = coords_data[y_col][coords_mask].values
        print(f"Coords data after zero removal: {len(coords_x)} positions")
    else:
        print("WARNING: Not enough columns in coords.csv")
        coords_x = np.array([])
        coords_y = np.array([])
    
except FileNotFoundError:
    print("WARNING: coords.csv file not found!")
    coords_x = np.array([])
    coords_y = np.array([])
except Exception as e:
    print(f"WARNING: Error reading coords.csv: {e}")
    coords_x = np.array([])
    coords_y = np.array([])

# Check if we have any valid data left
if len(coords_x) == 0:
    print("WARNING: No valid coords data after zero removal!")
    exit()

def myplot(x, y, s, bins=500, range_extent=None):
    if range_extent is None:
        range_extent = [[0, 130], [-3.5, 3.5]]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_extent)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

# Create heatmap from coords data
print(f"\nCreating heatmap...")
img_coords, extent = myplot(coords_x, coords_y, s=2)

print(f"\nHeatmap statistics:")
print(f"Coords frequency range: {np.min(img_coords):.4f} - {np.max(img_coords):.4f}")
print(f"Coords non-zero pixels: {np.sum(img_coords > 0)}")

# Check if heatmap has any significant values
if np.max(img_coords) < 0.1:
    print("WARNING: Heatmap has very low values!")
    print("This could mean:")
    print("1. Data points are sparse")
    print("2. Data is outside the range [0,130] x [-3.5,3.5]")
    print("3. Data has mostly zero coordinates")

# Create highly saturated custom colormap
colors_blue = ['#E6F3FF', '#99D6FF', '#4DB8FF', '#0080FF', '#0066CC', '#004C99']
cmap_blue = mcolors.LinearSegmentedColormap.from_list('saturated_blue', colors_blue, N=256)

# Use vmax to maximize contrast
vmax_coords = max(0.1, np.max(img_coords)) * 1.2
print(f"Using vmax_coords: {vmax_coords:.4f}")

# Create the single figure
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(img_coords, extent=extent, origin='lower', cmap=cmap_blue, 
              interpolation='nearest', aspect=10, vmax=vmax_coords)
ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
ax.grid(True, alpha=0.3)
ax.set_xlabel(r'$\it{x}$-coordinate (m)', fontsize=20)
ax.set_ylabel(r'$\it{y}$-coordinate (m)', fontsize=20)
ax.set_ylim([-3.5, 3.5])
ax.set_xlim([0, 130])
ax.set_facecolor('white')
ax.text(5, 0.25, 'Lane marking', dict(size=14), color='black',
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05, aspect=20)
cbar.set_label('Frequency', fontsize=18)
# Set integer ticks for colorbar
cbar.set_ticks(np.arange(0, int(vmax_coords) + 1, 1))

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig("coords_heatmap.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
plt.show()

print("\nDebugging complete!")
