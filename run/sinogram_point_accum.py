import numpy as np
import matplotlib.pyplot as plt

target_radius = 1.573e-2/2
antenna_radius = 21.0e-2 + 0.03618 + 0.0449
tar_x, tar_y = 3.0e-2, 0.

theta = np.flip(np.deg2rad(np.linspace(0, 355, 72) - 136.))

tar_xs = target_radius * np.cos(theta) + tar_x
tar_ys = target_radius * np.sin(theta) + tar_y

ant_xs = antenna_radius * np.cos(theta)
ant_ys = antenna_radius * np.sin(theta)

start = 100
end = 350

map = np.zeros([end-start, 72])

rho = np.sqrt((ant_xs[:, None] - tar_xs)**2 + (ant_ys[:, None] - tar_ys)**2)

surface_times = ((np.sqrt((ant_xs - tar_x)**2 + (ant_ys - tar_y)**2) -
                 target_radius) * 2 / 3e8) * 1e9

surface_times_behind = ((np.sqrt((ant_xs - tar_x)**2 + (ant_ys - tar_y)**2) +
                 target_radius) * 2 / 3e8) * 1e9

times = np.linspace(.5e-9, 5.5e-9, 700)
dists = np.tile((times * 3e8 / 2)[start:end].reshape(end-start, 1), 72)

for ant_pos in range(72):
    mask = np.isclose(dists, rho[:, ant_pos], atol=1e-3)
    values = 1 / rho[:, ant_pos] ** 2

    for col in range(72):
        map[mask[:, col], col] += values[col]

ts = np.linspace(0.5, 5.5, 700)
plt_extent = [0, 355, ts[end], ts[start]]
plt_aspect_ratio = 355 / ts[end]

plt.figure()
plt.rc('font', family='Times New Roman')
plt.imshow(map, cmap='inferno', aspect=plt_aspect_ratio, extent=plt_extent)
plt.colorbar(format='%.2e').ax.tick_params(labelsize=16)
plt.gca().set_yticks([round(ii, 2)
                              for ii in ts[start:end:np.size(ts) // 8]])
plt.ylabel('Time of Response (ns)', fontsize=16)
plt.gca().set_xticks([round(ii)
                          for ii in np.linspace(0, 355, 355)[::75]])
plt.title(f'Model target at ({tar_x*100:.1f},{tar_y})', fontsize=20)
plt.xlabel('Polar Angle of Antenna Position ('
           + r'$^\circ$' + ')',
           fontsize=16)
plt.plot(np.linspace(0, 355, 72), surface_times, linestyle='-',
         color='b', linewidth=2, label='Surface reflections (front)')
plt.plot(np.linspace(0, 355, 72), surface_times_behind, linestyle='--',
         color='b', linewidth=2, label='Surface reflections (back)')
plt.legend(fontsize=14, loc='lower right')
plt.tight_layout()

# plt.savefig('C:/Users/prikh/Desktop/MWIBC/UofM_MWIBC/output/cyl_phantom'
#             '/recons/Immediate '
#             'reference/20240109_glass_rod/sinograms/sino_4_with_surf.png',
#             dpi=300, transparent=True)




