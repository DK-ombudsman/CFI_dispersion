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
    file_names = [
    'omega_slice0_y128_stride1_branch-1.0.csv',
    'omega_slice_y128_stride1_branch-1.0.csv',
    'omega_slice2_y128_stride1_branch-1.0.csv',
    'omega_slice3_y128_stride1_branch-1.0.csv'
     ]
    file_names2 = [
    'omega_slice0_y128_mod_branch-1.0.csv',    
    'omega_slice_y128_mod_branch-1.0.csv', 
    'omega_slice2_y128_mod_branch-1.0.csv', 
    'omega_slice3_y128_mod_branch-1.0.csv', 
    ]
   



# Define axis limits
X_MIN, X_MAX = -150.0, 150.0
Z_MIN, Z_MAX = -100.0, 100.0

# Store all positive values for global color scaling
#all_positive_values = []
all_ratio_values= []
#all_x_coords = []
#all_z_coords = []

# Process each file first to collect data for color scaling
for i, (file_name, file_name2) in enumerate(zip(file_names, file_names2)):
    # Load the original data
    data = pd.read_csv(file_name)
    
    # Convert x and z from cm to km
    data['x_km'] = data['x(cm)'] / 100000
    data['z_km'] = data['z(cm)'] / 100000
    
    # Filter out non-finite values first
    data_clean = data[np.isfinite(data['omega_imag_cm^-1'])]

    # Separate positive and negative omega_imag values
    positive_omega = data_clean[data_clean['omega_imag_cm^-1'] > 0]

    #Load modified data
    data2 = pd.read_csv(file_name2)
    data2['x_km'] = data2['x(cm)'] / 100000
    data2['z_km'] = data2['z(cm)'] / 100000
    data_clean2 = data2[np.isfinite(data2['omega_imag_cm^-1'])]
    positive_omega2 = data_clean2[data_clean2['omega_imag_cm^-1'] > 0]
    

    # Create pivot tables for both datasets
    positive_pivot = positive_omega.pivot_table(values='omega_imag_cm^-1', index='z_km', columns='x_km', aggfunc='mean')
    positive_pivot2 = positive_omega2.pivot_table(values='omega_imag_cm^-1', index='z_km', columns='x_km', aggfunc='mean')

    x_unique = np.sort(data_clean['x_km'].unique())
    z_unique = np.sort(data_clean['z_km'].unique())

    # Reindex both pivot tables to the same grid
    positive_pivot_full = positive_pivot.reindex(index=z_unique, columns=x_unique) 
    positive_pivot_full2 = positive_pivot2.reindex(index=z_unique, columns=x_unique)
    
    # Get values
    original_values = positive_pivot_full.values
    modified_values = positive_pivot_full2.values

    # Calculate ratio where both exist
    ratio_matrix = np.full_like(original_values, np.nan)
    both_exist_mask = (~np.isnan(original_values)) & (~np.isnan(modified_values))
    ratio_matrix[both_exist_mask] = modified_values[both_exist_mask] / original_values[both_exist_mask]

    # Collect ratio values for global scaling
    valid_ratios = ratio_matrix[both_exist_mask]
    if len(valid_ratios) > 0:
        all_ratio_values.extend(valid_ratios)


# Calculate global color scaling
all_ratio_values = np.array(all_ratio_values)
vmin, vmax = float(np.min(all_ratio_values)), float(np.max(all_ratio_values))

# Use log scaling if dynamic range is large, otherwise use linear with percentile clipping
if vmin > 0 and (np.log10(vmax) - np.log10(vmin) > 3):
    norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    # Use percentile clipping for linear scaling to avoid outliers
    q2, q98 = np.percentile(all_ratio_values, [2, 98])
    if q2 < q98:
        vmin, vmax = q2, q98
    norm = None

set_manual_scale= True

if set_manual_scale:
    V_MIN = 0.1  # Adjust this value
    V_MAX = 10 # Adjust this value

    # Use log scaling for the manually set range
    norm = LogNorm(vmin=V_MIN, vmax=V_MAX)

cmap_status = ListedColormap(['red', 'orange'])

# Process and plot each file
for i, (file_name, file_name2) in enumerate(zip(file_names, file_names2)):
    # Load the data
    data = pd.read_csv(file_name)
    
    # Convert x and z from cm to km
    data['x_km'] = data['x(cm)'] / 100000
    data['z_km'] = data['z(cm)'] / 100000

    # Filter out non-finite values first
    data_clean = data[np.isfinite(data['omega_imag_cm^-1'])]
    
    # Separate positive and negative omega_imag values
    positive_omega = data_clean[data_clean['omega_imag_cm^-1'] > 0]

    #Load modified data
    data2 = pd.read_csv(file_name2)
    data2['x_km'] = data2['x(cm)'] / 100000
    data2['z_km'] = data2['z(cm)'] / 100000
    data_clean2 = data2[np.isfinite(data2['omega_imag_cm^-1'])]
    positive_omega2 = data_clean2[data_clean2['omega_imag_cm^-1'] > 0]
    

    # Create pivot tables for both data sets
    positive_pivot = positive_omega.pivot_table(values='omega_imag_cm^-1', index= 'z_km', columns='x_km', aggfunc='mean')
    positive_pivot2 = positive_omega2.pivot_table(values='omega_imag_cm^-1', index='z_km', columns='x_km', aggfunc='mean')


    x_unique = np.sort(data_clean['x_km'].unique())
    z_unique = np.sort(data_clean['z_km'].unique())

    # Reindex both pivot tables to the same grid
    positive_pivot_full = positive_pivot.reindex(index=z_unique, columns=x_unique) 
    positive_pivot_full2 = positive_pivot2.reindex(index=z_unique, columns=x_unique) 
    
    
    X, Z = np.meshgrid(x_unique, z_unique)

    
    # Get values
    original_values = positive_pivot_full.values
    modified_values = positive_pivot_full2.values

    # Calculate ratio where both exist
    ratio_matrix = np.full_like(original_values, np.nan)
    both_exist_mask = (~np.isnan(original_values)) & (~np.isnan(modified_values))
    ratio_matrix[both_exist_mask] = modified_values[both_exist_mask] / original_values[both_exist_mask]
    
    status_matrix = np.full_like(original_values, np.nan)
    # New positive growth rate (present in modified but not in original)
    new_mask = (np.isnan(original_values)) & (~np.isnan(modified_values))
    status_matrix[new_mask] = 1  # Blue for new
    
    # Vanished positive growth rate (present in original but not in modified)
    vanished_mask = (~np.isnan(original_values)) & (np.isnan(modified_values))
    status_matrix[vanished_mask] = 0  # Red for vanished
    
    # Plot the heatmap using pcolormesh with global color scaling
    if norm is not None:    
        im_ratio = plt.pcolormesh(X, Z, ratio_matrix, cmap='viridis', shading='auto', norm=norm)
       
    else:
        # Use vmin/vmax (for linear scaling)
        
        im_ratio = plt.pcolormesh(X, Z, ratio_matrix, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        
    
    im_status = plt.pcolormesh(X, Z, status_matrix, cmap=cmap_status, shading='auto', vmin=0, vmax=1)
  
        
# Add colorbar for ratios
cbar_ratio = plt.colorbar(im_ratio, label='Ratio (Modified/Original)')
#cbar_ratio.set_ticks([0.1, 0.5, 1, 2, 10])
#cbar_ratio.set_ticklabels(['0.1', '0.5', '1', '2', '10'])

# Add colorbar for status
cbar_status = plt.colorbar(im_status, label='New/vanished growth rates')
cbar_status.set_ticks([0.25, 0.75])
cbar_status.set_ticklabels(['Vanished', 'New'])


plt.xlabel('x (km)')
plt.ylabel('z (km)')

if mono:
    plt.title('Positive I($\\omega$) at y=128 slice mono combined')
else:
    plt.title('Growth rate comparison at y=128 slice multi combined branch=-1')

# Set axis limits
plt.xlim(X_MIN, X_MAX)
plt.ylim(Z_MIN, Z_MAX)

plt.tight_layout()

if mono:
    out = "combined_mono_plots.jpeg"
else:
    out = "comparison.jpeg"

plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved to {out}")