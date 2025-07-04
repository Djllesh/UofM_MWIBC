import numpy as np
import multiprocessing as mp
from umbms.beamform.time_delay import get_pix_ts
from umbms.plot.imgplots import plot_fd_img_with_intersections
from umbms.beamform.utility import get_ant_scan_xys, get_xy_arrs

__CPU_COUNT = mp.cpu_count()

if __name__ == "__main__":
    worker_pool = mp.Pool(__CPU_COUNT - 1)

    ant_rad = 0.21
    m_size = 150
    air_speed = 3e8
    breast_speed = 2e8
    adi_rad = 0.055
    roi_rad = adi_rad + 0.01

    img = np.zeros([m_size, m_size])

    pix_xs, pix_ys = get_xy_arrs(m_size, roi_rad)
    ant_pos_xs, ant_pos_ys = get_ant_scan_xys(ant_rad=ant_rad, n_ant_pos=72)

    pix_ts, int_f_xs, int_f_ys, int_b_xs, int_b_ys = get_pix_ts(
        ant_rad=ant_rad,
        m_size=m_size,
        roi_rad=roi_rad,
        air_speed=air_speed,
        breast_speed=breast_speed,
        adi_rad=adi_rad,
        worker_pool=worker_pool,
    )

    plot_fd_img_with_intersections(
        img=img,
        roi_rad=roi_rad,
        img_rad=roi_rad,
        ant_pos_x=ant_pos_xs[0],
        ant_pos_y=ant_pos_ys[0],
        pix_xs=pix_xs[0, :],
        pix_ys=pix_ys[:, 0],
        int_f_xs=int_f_xs,
        int_f_ys=int_f_ys,
        int_b_xs=int_b_xs,
        int_b_ys=int_b_ys,
        adi_rad=adi_rad,
    )

    worker_pool.close()
