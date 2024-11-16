"""
Illia Prykhodko
University of Manitoba
October 16th 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import noise
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

# Function to get binary mask based on cubic
# spline interpolation (similar to provided logic)
def get_binary_mask(cs, m_size, roi_rad, precision_scaling_factor=1):
    """Returns a binary mask that corresponds to a
    boundary using cubic spline interpolation."""
    pix_xs, pix_ys = get_xy_arrs(m_size=m_size * precision_scaling_factor,
                                 roi_rad=roi_rad)
    rho = np.sqrt(pix_xs ** 2 + pix_ys ** 2)
    phi = np.arctan2(pix_ys, pix_xs)
    phi[phi < 0] += 2 * np.pi  # Adjust angles to [0, 2pi]
    rho_fit = cs(phi)
    mask = rho <= rho_fit  # Mask points inside the boundary
    return mask

# Generate grid of points (x, y) in a square domain
def get_xy_arrs(m_size, roi_rad):
    x_vals = np.linspace(-roi_rad, roi_rad, m_size)
    y_vals = np.linspace(-roi_rad, roi_rad, m_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    return X, Y

# Parameters for the circle
radius = 1
theta = np.linspace(0, 2 * np.pi, 500)  # 500 points for a smooth circle

# Function to generate a noisy circle
def add_perlin_noise_2d(theta, noise_strength=0.05, scale=5.0):
    perlin_noise = np.array(
        [noise.pnoise2(np.cos(t) * scale, np.sin(t) * scale) for t in theta])
    return np.ones_like(theta) + noise_strength * perlin_noise

# Generate a perfect circle
def generate_circle(radius, theta):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


fig = plt.figure(constrained_layout=True, figsize=(14, 7))
subfig = fig.subfigures()
# Create a 2x3 grid of subplots with a shared colorbar
axes = subfig.subplots(2, 3)

# Shared color map
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=0.05)

# Remove ticks and labels for all axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Plot the original perfect circle in the top-left plot
x_circle, y_circle = generate_circle(radius, theta)
axes[0, 0].set_facecolor((68/256, 1/256, 84/256, 0.4))
axes[0, 0].set_aspect('equal', 'box')
axes[0, 0].set_title('Uniform propagation speed')
axes[0, 0].text(s=r'$v = v_{avg}$', x=0.5 - 0.05, y=0.5)
# Plot 4 mishapen circles with Perlin noise
for i in range(2):
    for j in range(3):
        ax = axes[i, j]
        if i == 0 and j == 0:
            # Skip the first plot, already filled with the perfect circle
            continue
        if i == 1 and j == 2:
            # Skip the last circle for now, it will be filled with noise
            continue
        if i == 0 and j == 1:

            ax.fill(x_circle, y_circle, color=cmap(norm(0.01)),
                  edgecolor='k', lw=1.7, alpha=0.6)
            ax.set_aspect('equal', 'box')
            ax.set_title(f'Binary Modelling')
            ax.text(s=r'$v_{in}$', x=-0.05, y=0)
            ax.text(s=r'$v_{out} = c$', x=0.446, y=-1)
            continue
        if i == 0 and j == 2:

            # Plot a few transparent, filled layers of mishapen circles with a slight shift
            n_circles = 15
            alpha = 0.3  # Transparency for each circle
            shift_step = 0.02  # Amount of shift to the top-right for each circle

            for ii in range(n_circles):
                noise_strength = 0.02 + 0.001 * ii
                # Shift the circle slightly to the top-right
                shift_x = (n_circles - ii) * shift_step
                shift_y = (n_circles - ii) * shift_step

                x_plt = radius * np.cos(theta) + shift_x
                y_plt = radius * np.sin(theta) + shift_y

                # Use fill to create a filled, transparent circle with varying color
                ax.fill(x_plt, y_plt, color=cmap(norm(noise_strength)),
                        alpha=0.4, edgecolor='k')

            # Set aspect and limits to match the previous plot
            ax.set_aspect('equal')
            ax.annotate(text='', xy=(0.866, -0.646), xytext=(1.329, -0.153),
                         arrowprops=dict(arrowstyle='<->'))
            ax.text(s='1001', x=1.064, y=-0.576)
            ax.set_title('Frequency-dependent modelling')
            ax.text(s=r'$v_{in}(f)$', x=-0.05, y=0)
            ax.text(s=r'$v_{out} = c$', x=0.446, y=-1)
            continue
        if i == 1 and j == 1:

            # Plot a few transparent, filled layers of mishapen circles with a slight shift
            n_circles = 15
            alpha = 0.3  # Transparency for each circle
            shift_step = 0.02  # Amount of shift to the top-right for each circle

            for ii in range(n_circles):
                noise_strength = 0.02 + 0.0002 * ii  # Vary noise strength
                # slightly for variety
                r_noise = add_perlin_noise_2d(theta,
                                              noise_strength=noise_strength
                                                             + 0.05,
                                              scale=3.0)

                # Shift the circle slightly to the top-right
                shift_x = (n_circles - ii) * shift_step
                shift_y = (n_circles - ii) * shift_step

                x_mishapen = r_noise * np.cos(theta) + shift_x
                y_mishapen = r_noise * np.sin(theta) + shift_y

                # Use fill to create a filled, transparent circle with varying color
                ax.fill(x_mishapen, y_mishapen,
                        color=cmap(norm(noise_strength)),
                        alpha=0.4, edgecolor='k')

            # Set aspect and limits to match the previous plot
            ax.set_aspect('equal')
            ax.annotate(text='', xy=(0.862, -0.659),
                        xytext=(1.178, -0.395),
                        arrowprops=dict(arrowstyle='<->'))
            ax.text(s='1001', x=1.007, y=-0.65)
            ax.text(s=r'$v_{in}(f)$', x=-0.05, y=0)
            ax.text(s=r'$v_{out} = c$', x=0.446, y=-1)
            ax.set_title('Frequency-dependent modelling (boundary detection)')
            continue

        # Add Perlin noise with varying noise_strength for variety
        noise_strength = 0.01 * (i * 3 + j)
        radius_noise = 0.04
        # Vary noise strength for different circles
        r_noise = add_perlin_noise_2d(theta, noise_strength=radius_noise,
                                      scale=3.0)
        x_mishapen = r_noise * np.cos(theta)
        y_mishapen = r_noise * np.sin(theta)
        ax.fill(x_mishapen, y_mishapen, color=cmap(norm(noise_strength)),
                edgecolor='k', lw=1.7, alpha=0.6)
        ax.set_aspect('equal', 'box')
        ax.set_title('Boundary detection, binary modelling')
        ax.text(s=r'$v_{in}$', x=-0.05, y=0)
        ax.text(s=r'$v_{out} = c$', x=0.446, y=-1)

# Get the limits from one of the previous plots (e.g., the second one)
xlim = axes[0, 1].get_xlim()
ylim = axes[0, 1].get_ylim()
# Fill the last mishapen circle with Perlin noise using cubic spline interpolation
ax_last = axes[1, 2]
noise_strength = 0.04  # Choose a noise strength for the last circle
r_noise_last = add_perlin_noise_2d(theta, noise_strength=noise_strength,
                                   scale=3.0)
x_mishapen_last = r_noise_last * np.cos(theta)
y_mishapen_last = r_noise_last * np.sin(theta)

# Interpolate using cubic spline
cs_r = CubicSpline(theta, r_noise_last, bc_type='periodic')

# Create binary mask
m_size = 1500  # Resolution of the mask
mask = get_binary_mask(cs_r, m_size=m_size,
                    roi_rad=np.abs(xlim[0] - xlim[1])/2)

# Generate Perlin noise to fill inside the mask
x_vals, y_vals = get_xy_arrs(m_size, roi_rad=radius)
perlin_grid = np.full((m_size, m_size), np.nan)
for i in range(m_size):
    for j in range(m_size):
        perlin_grid[i, j] = noise.pnoise2(x_vals[i, j] * 5, y_vals[i, j] * 5)


# Normalize noise for colormap
perlin_grid_normalized = norm(perlin_grid)
perlin_grid_normalized[np.logical_not(mask)] = np.NaN
# Set the extent based on the extracted limits
extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

# Plot the last circle filled with Perlin noise
im = ax_last.imshow(perlin_grid_normalized, extent=extent, cmap='viridis',
               origin='lower')

# Plot the outline of the mishapen circle on top
ax_last.plot(x_mishapen_last * 0.99, y_mishapen_last * 0.99, color='k', lw=1.7)
ax_last.set_aspect('equal', 'box')
ax_last.set_title('Extracted Propagation Speed, Boundary '
                  'Detection')

# Set the same xlim and ylim for the last plot as the others
ax_last.set_xlim(extent[0], extent[1])
ax_last.set_ylim(extent[2], extent[3])

# cbar = subfig.colorbar(im, ax=axes, location='right')
# # Add a shared colorbar to the right of all subplots
# # cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
# #                     location='right', shrink=0.8)
# cbar.set_label('Noise Strength')

# Display the plot
plt.show()
