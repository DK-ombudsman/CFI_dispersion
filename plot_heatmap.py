import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap


# Create a figure
plt.figure(figsize=(10, 8))

# Define the file names
mono= False
if mono:
    file_names = [
    'omega_slice0_y128_stride1_mono.csv',
    'omega_slice_y128_stride1_mono.csv',
    'omega_slice2_y128_stride1_mono.csv',
    'omega_slice3_y128_stride1_mono.csv'
     ]
else:
    file_names2 = [
    'omega_slice0_y128_stride1_branch-1.0.csv',
    'omega_slice_y128_stride1_branch-1.0.csv',
    'omega_slice2_y128_stride1_branch-1.0.csv',
    'omega_slice3_y128_stride1_branch-1.0.csv'
     ]
    file_names = [
    'omega_slice0_y128_mod_branch-1.0.csv',    
    'omega_slice_y128_mod_branch-1.0.csv', 
    'omega_slice2_y128_mod_branch-1.0.csv', 
    'omega_slice3_y128_mod_branch-1.0.csv', 
    ]
   



# Define axis limits
X_MIN, X_MAX = -150.0, 150.0
Z_MIN, Z_MAX = -80.0, 80.0

# Store all positive values for global color scaling
all_positive_values = []

all_x_coords = []
all_z_coords = []

# Process each file first to collect data for color scaling
for i, file_name in enumerate(file_names):
    # Load the data
    data = pd.read_csv(file_name)
    
    # Convert x and z from cm to km
    data['x_km'] = data['x(cm)'] / 100000
    data['z_km'] = data['z(cm)'] / 100000
    
    # Filter out non-finite values first
    data_clean = data[np.isfinite(data['omega_imag_cm^-1'])]

    # Separate positive and negative omega_imag values
    positive_omega = data_clean[data_clean['omega_imag_cm^-1'] > 0]
    #negative_omega = data_clean[data_clean['omega_imag_cm^-1'] <= 0]

    # Collect all coordinates
    all_x_coords.extend(data_clean['x_km'].values)
    all_z_coords.extend(data_clean['z_km'].values)
    
    # Collect values for global color scaling
    all_positive_values.extend(positive_omega['omega_imag_cm^-1'].values)

# Calculate global color scaling
all_positive_values = np.array(all_positive_values)
vmin, vmax = float(np.min(all_positive_values)), float(np.max(all_positive_values))

# Use log scaling if dynamic range is large, otherwise use linear with percentile clipping
if vmin > 0 and (np.log10(vmax) - np.log10(vmin) > 3):
    norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    # Use percentile clipping for linear scaling to avoid outliers
    q2, q98 = np.percentile(all_positive_values, [2, 98])
    if q2 < q98:
        vmin, vmax = q2, q98
    norm = None

set_manual_scale= True

if set_manual_scale:
    V_MIN = 1e-11  # Adjust this value
    V_MAX = 1e2  # Adjust this value

    # Use log scaling for the manually set range
    norm = LogNorm(vmin=V_MIN, vmax=V_MAX)



# Create unified grid
x_unique = np.sort(np.unique(all_x_coords))
z_unique = np.sort(np.unique(all_z_coords))

# Create edge coordinates for pcolormesh
dx = np.diff(x_unique)
dz = np.diff(z_unique)
xe = np.r_[x_unique[0] - dx[0]/2, x_unique[:-1] + dx/2, x_unique[-1] + dx[-1]/2]
ze = np.r_[z_unique[0] - dz[0]/2, z_unique[:-1] + dz/2, z_unique[-1] + dz[-1]/2]
Xe, Ze = np.meshgrid(xe, ze)

#X, Z = np.meshgrid(x_unique, z_unique)

# Process and plot each file
for i, file_name in enumerate(file_names):
    # Load the data
    data = pd.read_csv(file_name)
    
    # Convert x and z from cm to km
    data['x_km'] = data['x(cm)'] / 100000
    data['z_km'] = data['z(cm)'] / 100000

    # Filter out non-finite values first
    data_clean = data[np.isfinite(data['omega_imag_cm^-1'])]
    
    # Separate positive and negative omega_imag values
    positive_omega = data_clean[data_clean['omega_imag_cm^-1'] > 0]
    negative_omega = data_clean[data_clean['omega_imag_cm^-1'] <= 0]

    # Create pivot tables for both positive and negative values
    positive_pivot = positive_omega.pivot_table(values='omega_imag_cm^-1', index= 'z_km', columns='x_km', aggfunc='mean')
    negative_pivot = negative_omega.pivot_table(values='omega_imag_cm^-1', index='z_km', columns='x_km', aggfunc='mean')

    x_unique = np.sort(data_clean['x_km'].unique())
    z_unique = np.sort(data_clean['z_km'].unique())

    # Reindex both pivot tables to the same grid
    positive_pivot_full = positive_pivot.reindex(index=z_unique, columns=x_unique) 
    negative_pivot_full = negative_pivot.reindex(index=z_unique, columns=x_unique)
    
    X, Z = np.meshgrid(x_unique, z_unique)

    # Get values
    positive_values = positive_pivot_full.values
    negative_values = negative_pivot_full.values

    # Create mask for negative values
    negative_mask = ~np.isnan(negative_values)


    # Create a separate array for negative values
    negative_display = np.zeros_like(negative_values)
    negative_display[negative_mask] = 1  # Mark negative value positions with 1

    # Plot the negative values as white
    white_cmap = ListedColormap(['none', 'none'])  # 0 -> transparent, 1 -> white
    

    
    # Plot the heatmap using pcolormesh with global color scaling
    if norm is not None:
        # Use norm (for log scaling)
        # First plot positive values with colormap
        im_positive = plt.pcolormesh(X, Z, positive_values, cmap='viridis', shading='auto', norm=norm)
        # Plot the negative values as white
        im_negative = plt.pcolormesh(X, Z, negative_display, cmap=white_cmap, shading='auto', norm=norm)

    else:
        # Use vmin/vmax (for linear scaling)
        # First plot positive values with colormap
        im_positive = plt.pcolormesh(X, Z, positive_values, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        # Plot the negative values as white
        im_negative = plt.pcolormesh(X, Z, negative_display, cmap=white_cmap, shading='auto', vmin=vmin, vmax=vmax)

# Add colorbar (only for positive values) and labels
cbar = plt.colorbar(im_positive, label='I($\\omega$) (cm^-1) (positive)')
plt.xlabel('x (km)')
plt.ylabel('z (km)')

if mono:
    plt.title('Positive I($\\omega$) at y=128 slice mono combined')
else:
    plt.title('Positive I($\\omega$) at y=128 slice multi combined branch=-1')

# Set axis limits
plt.xlim(X_MIN, X_MAX)
plt.ylim(Z_MIN, Z_MAX)

plt.tight_layout()

if mono:
    out = "combined_mono_plots.jpeg"
else:
    out = "mod_combined_multi_plots.jpeg"

plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved to {out}")