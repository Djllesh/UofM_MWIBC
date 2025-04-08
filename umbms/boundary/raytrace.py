"""
Illia Prykhodko
University of Manitoba
January 24th, 2023
"""

from umbms.beamform.utility import get_ant_scan_xys
import numpy as np
from functools import partial


###############################################################################


def find_boundary_rt(
    binary_mask,
    ant_rad,
    roi_rad,
    *,
    n_ant_pos=72,
    ini_ant_ang=-136.0,
    precision_scaling_factor=1,
    worker_pool=None,
):
    """Finds the intersection points of a given shape analogous to the
     circular or elliptical approximations

    *Assumed that the pixels are square and so is the domain*

    Parameters:
    ------------
    binary_mask : array_like MxM
        Boolean mask that specifies the postition of each pixel with
        respect to the phantom boundary (in or out)
    n_ant_pos : int
        Number of antenna posistions during a scan
    ant_rad : float
        Corrected antenna radius that was used during a scan
    roi_rad : float
        Radius of region of interest, determines the domain size
    ini_ant_and : float
        Angle of initial antenna rotation during the scan
    precision_scaling_factor : int
        Scaling factor for the image domain (for more accurate rt)
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns:
    ------------
    int_f_xs, int_f_ys, int_b_xs, int_b_ys : array_like
        Arrays of front and back intersections for all antenna positions
        (The unintersected pixels follow the same logic as in circular
        or elliptic approximations)
    """

    # binary mask is a square array that correspond to the spacial
    # extent of the domain thus the size of either side is the exact
    # size of the domain in pixels

    # divide by scaling factor to obtain image size at the end
    m_size = np.size(binary_mask, axis=0) // precision_scaling_factor
    # for finer grid
    m_size_pixels = np.size(binary_mask, axis=0)

    # initializing arrays for storing intersection coordinates
    # front intersection - closer to antenna
    int_f_xs = np.zeros([n_ant_pos, m_size, m_size], dtype=float)
    int_f_ys = np.zeros_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.zeros_like(int_f_xs)
    int_b_ys = np.zeros_like(int_f_xs)

    # obtain antenna and pixel coordinates
    ant_xs, ant_ys = get_ant_scan_xys(
        ant_rad=ant_rad, n_ant_pos=n_ant_pos, ini_ant_ang=ini_ant_ang
    )
    pix_xs = np.linspace(-roi_rad, roi_rad, m_size_pixels)
    pix_ys = np.flip(pix_xs)
    d = pix_xs[1] - pix_xs[0]

    # initialize planes
    x_planes = np.linspace(
        -roi_rad, -roi_rad + d * m_size_pixels, m_size_pixels + 1
    )
    y_planes = x_planes

    if worker_pool is not None:
        iterable_idx = range(n_ant_pos * m_size**2)

        parallel_func = partial(
            parallel_intersections_per_ant_pos_rt,
            binary_mask,
            ant_xs,
            ant_ys,
            pix_xs,
            pix_ys,
            x_planes,
            y_planes,
            precision_scaling_factor,
            d,
        )

        intersections = np.array(worker_pool.map(parallel_func, iterable_idx))
        intersection_shape = [n_ant_pos, m_size, m_size]
        int_f_xs[:, :, :] = np.reshape(intersections[:, 0], intersection_shape)
        int_f_ys[:, :, :] = np.reshape(intersections[:, 1], intersection_shape)
        int_b_xs[:, :, :] = np.reshape(intersections[:, 2], intersection_shape)
        int_b_ys[:, :, :] = np.reshape(intersections[:, 3], intersection_shape)

    else:
        for ant_pos in range(n_ant_pos):
            (
                int_f_xs[ant_pos, :, :],
                int_f_ys[ant_pos, :, :],
                int_b_xs[ant_pos, :, :],
                int_b_ys[ant_pos, :, :],
            ) = intersections_per_antenna_pos_rt(
                binary_mask,
                ant_xs[ant_pos],
                ant_ys[ant_pos],
                pix_xs,
                pix_ys,
                x_planes,
                y_planes,
                d,
                precision_scaling_factor,
            )

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def intersections_per_antenna_pos_rt(
    binary_mask,
    ant_pos_x,
    ant_pos_y,
    pix_xs,
    pix_ys,
    x_planes,
    y_planes,
    d,
    precision_scaling_factor,
):
    """Returns a set of front and back intersections arrays that
    correspond to an antenna position and to a binary mask

    Parameters:
    -----------
    binary_mask : array_like
        Boolean mask that specifies the postition of each pixel with
        respect to the phantom boundary (in or out)
    ant_pos_x : float
        x_coord of the antenna position
    ant_pos_y : float
        y_coord of the antenna position
    pix_xs : array_like
        An array of x_coords of pixels in the domain
    pix_ys : array_like
        An array of y_coords of pixels in the domain
    x_planes : array_like
        An array of spacial positions of orthogonal planes (vertical)
    y_planes : array_like
        An array of spacial positions of orthogonal planes (horizontal)
    d : float
        Distance between planes and the size of the pixel
    precision_scaling_factor : int
        Scaling factor for the image domain (for more accurate rt)

    Returns:
    int_f_xs, int_f_ys, int_b_xs, int_b_ys : array_like
        Arrays of front and back intersections for the antenna position
        (The unintersected pixels follow the same logic as in circular
        or elliptic approximations)
    """

    m_size = np.size(binary_mask, axis=0) // precision_scaling_factor

    int_f_xs = np.zeros([m_size, m_size], dtype=float)
    int_f_ys = np.zeros_like(int_f_xs)
    int_b_xs = np.zeros_like(int_f_xs)
    int_b_ys = np.zeros_like(int_f_xs)

    for x_idx_no_scale in range(m_size):
        for y_idx_no_scale in range(m_size):
            x_idx = x_idx_no_scale * precision_scaling_factor
            y_idx = y_idx_no_scale * precision_scaling_factor

            alpha_x = np.array([])
            alpha_y = np.array([])

            alpha_x_0 = (x_planes[0] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)
            alpha_x_n = (x_planes[-1] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)

            alpha_y_0 = (y_planes[0] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)
            alpha_y_n = (y_planes[-1] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)

            alpha_min = max(
                0.0, min(alpha_x_0, alpha_x_n), min(alpha_y_0, alpha_y_n)
            )

            alpha_max = min(
                1.0, max(alpha_x_0, alpha_x_n), max(alpha_y_0, alpha_y_n)
            )

            if pix_xs[x_idx] - ant_pos_x > 0:
                i_min = np.size(x_planes) - np.floor(
                    (
                        x_planes[-1]
                        - alpha_min * (pix_xs[x_idx] - ant_pos_x)
                        - ant_pos_x
                    )
                    / d
                ).astype("int")

                i_max = 1 + np.floor(
                    (
                        ant_pos_x
                        + alpha_max * (pix_xs[x_idx] - ant_pos_x)
                        - x_planes[0]
                    )
                    / d
                ).astype("int")

                if i_min >= 0 and i_max >= 0:
                    i_set = np.arange(i_min, i_max)
                    alpha_x = (x_planes[i_set] - ant_pos_x) / (
                        pix_xs[x_idx] - ant_pos_x
                    )

            else:
                i_min = np.size(x_planes) - np.floor(
                    (
                        x_planes[-1]
                        - alpha_max * (pix_xs[x_idx] - ant_pos_x)
                        - ant_pos_x
                    )
                    / d
                ).astype("int")

                i_max = 1 + np.floor(
                    (
                        ant_pos_x
                        + alpha_min * (pix_xs[x_idx] - ant_pos_x)
                        - x_planes[0]
                    )
                    / d
                ).astype("int")

                if i_min >= 0 and i_max >= 0:
                    i_set = np.arange(i_min, i_max)
                    i_set = np.flip(i_set)
                    alpha_x = (x_planes[i_set] - ant_pos_x) / (
                        pix_xs[x_idx] - ant_pos_x
                    )

            if pix_ys[y_idx] - ant_pos_y > 0:
                j_min = np.size(y_planes) - np.floor(
                    (
                        y_planes[-1]
                        - alpha_min * (pix_ys[y_idx] - ant_pos_y)
                        - ant_pos_y
                    )
                    / d
                ).astype("int")

                j_max = 1 + np.floor(
                    (
                        ant_pos_y
                        + alpha_max * (pix_ys[y_idx] - ant_pos_y)
                        - y_planes[0]
                    )
                    / d
                ).astype("int")

                if j_min >= 0 and j_max >= 0:
                    j_set = np.arange(j_min, j_max)
                    alpha_y = (y_planes[j_set] - ant_pos_y) / (
                        pix_ys[y_idx] - ant_pos_y
                    )

            else:
                j_min = np.size(y_planes) - np.floor(
                    (
                        y_planes[-1]
                        - alpha_max * (pix_ys[y_idx] - ant_pos_y)
                        - ant_pos_y
                    )
                    / d
                ).astype("int")

                j_max = 1 + np.floor(
                    (
                        ant_pos_y
                        + alpha_min * (pix_ys[y_idx] - ant_pos_y)
                        - y_planes[0]
                    )
                    / d
                ).astype("int")

                if j_min >= 0 and j_max >= 0:
                    j_set = np.arange(j_min, j_max)
                    j_set = np.flip(j_set)
                    alpha_y = (y_planes[j_set] - ant_pos_y) / (
                        pix_ys[y_idx] - ant_pos_y
                    )

            alpha_no_minmax = np.unique(np.concatenate((alpha_x, alpha_y)))

            if alpha_no_minmax.size != 0:
                alpha = np.unique(
                    np.concatenate(
                        (
                            np.array([alpha_min]),
                            alpha_no_minmax,
                            np.array([alpha_max]),
                        )
                    )
                )
            else:
                alpha = np.unique(np.array([alpha_min, alpha_max]))

            alpha = alpha[(alpha >= 0.0) & (alpha <= 1.0)]

            alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])

            intersected_x_idxs = np.floor(
                (
                    ant_pos_x
                    + alpha_mid * (pix_xs[x_idx] - ant_pos_x)
                    - x_planes[0]
                )
                / d
            ).astype("int")

            intersected_y_idxs = np.floor(
                (
                    ant_pos_y
                    + alpha_mid * (pix_ys[y_idx] - ant_pos_y)
                    - y_planes[0]
                )
                / d
            ).astype("int")

            crop_x = (intersected_x_idxs >= 0) & (
                intersected_x_idxs < np.size(binary_mask, axis=0)
            )
            intersected_x_idxs = intersected_x_idxs[crop_x]
            intersected_y_idxs = intersected_y_idxs[crop_x]

            crop_y = (intersected_y_idxs >= 0) & (
                intersected_y_idxs < np.size(binary_mask, axis=0)
            )
            intersected_x_idxs = intersected_x_idxs[crop_y]
            intersected_y_idxs = intersected_y_idxs[crop_y]

            intersected_y_idxs = (
                m_size * precision_scaling_factor - 1 - intersected_y_idxs
            )

            (
                int_f_xs[y_idx_no_scale, x_idx_no_scale],
                int_f_ys[y_idx_no_scale, x_idx_no_scale],
                int_b_xs[y_idx_no_scale, x_idx_no_scale],
                int_b_ys[y_idx_no_scale, x_idx_no_scale],
            ) = intersections_per_ray(
                binary_mask,
                intersected_x_idxs,
                intersected_y_idxs,
                pix_xs,
                pix_ys,
                pix_xs[x_idx],
                pix_ys[y_idx],
            )

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def intersections_per_ray(
    binary_mask,
    intersected_x_idxs,
    intersected_y_idxs,
    pix_xs,
    pix_ys,
    pix_x,
    pix_y,
):
    """Returns coords of two points that correspond to intersections
    and follow the same logic as circular or elliptical approx

    Parameters:
    -----------
    binary_mask : array_like
        Boolean mask that specifies the postition of each pixel with
        respect to the phantom boundary (in or out)
    intersected_x_idxs : array_like
        A list of x_indicies of pixels that were intersected with a ray
    intersected_y_idxs : array_like
        A list of y_indicies of pixels that were intersected with a ray
    pix_xs : array_like
        An array of x_coords of pixels in the domain
    pix_ys : array_like
        An array of y_coords of pixels in the domain
    pix_x : float
        X coordinate of a pixel that indicates the end of the ray
    pix_y : float
        Y coordinate of a pixel that indicates the end of the ray
    Returns:
    -----------
    int_f_x, int_f_y, int_b_x, int_b_y : float
        Coordinates of front and back intersections
    """

    int_f_x = np.nan
    int_f_y = np.nan
    int_b_x = np.nan
    int_b_y = np.nan

    if intersected_x_idxs.size == 0:  # if no intersection
        # all intersection coords are pixel coords
        int_f_x = pix_x
        int_f_y = pix_y
        int_b_x = pix_x
        int_b_y = pix_y

        return int_f_x, int_f_y, int_b_x, int_b_y

    if binary_mask[intersected_y_idxs[-1], intersected_x_idxs[-1]]:
        # if the ray ends in the phantom
        # back intersections - pix coords
        int_b_x = pix_x
        int_b_y = pix_y

    # sequence of indices that correspond to pixels that are inside the
    # phantom
    binary_ray = binary_mask[intersected_y_idxs, intersected_x_idxs]
    inside_idxs = np.argwhere(binary_ray).flatten()

    if inside_idxs.size != 0:
        # front intersection is the algebraic average of two points -
        # one before the intersection and one after
        int_f_x = (
            pix_xs[intersected_x_idxs[inside_idxs[0]]]
            + pix_xs[intersected_x_idxs[inside_idxs[0]] - 1]
        ) / 2
        int_f_y = (
            pix_ys[intersected_y_idxs[inside_idxs[0]]]
            + pix_ys[intersected_y_idxs[inside_idxs[0]] - 1]
        ) / 2

        if np.isnan(int_b_x):
            # same for back intersections
            # (if they aren't already assigned)
            int_b_x = (
                pix_xs[intersected_x_idxs[inside_idxs[-1]]]
                + pix_xs[intersected_x_idxs[inside_idxs[-1]] + 1]
            ) / 2
            int_b_y = (
                pix_ys[intersected_y_idxs[inside_idxs[-1]]]
                + pix_ys[intersected_y_idxs[inside_idxs[-1]] + 1]
            ) / 2

    else:
        int_f_x = pix_x
        int_f_y = pix_y
        int_b_x = pix_x
        int_b_y = pix_y

    return int_f_x, int_f_y, int_b_x, int_b_y


def parallel_intersections_per_ant_pos_rt(
    binary_mask,
    ant_xs,
    ant_ys,
    pix_xs,
    pix_ys,
    x_planes,
    y_planes,
    precision_scaling_factor,
    d,
    idx,
):
    """Returns a set of front and back intersections arrays that
    correspond to an antenna position and to a binary mask (flattened)

    Parameters:
    -----------
    binary_mask : array_like
        Boolean mask that specifies the postition of each pixel with
        respect to the phantom boundary (in or out)
    ant_xs : array_like
        An array of x_coords of antenna pos in the domain
    ant_ys : array_like
        An array of y_coords of antenna pos in the domain
    pix_xs : array_like
        An array of x_coords of pixels in the domain
    pix_ys : array_like
        An array of y_coords of pixels in the domain
    x_planes : array_like
        An array of spacial positions of orthogonal planes (vertical)
    y_planes : array_like
        An array of spacial positions of orthogonal planes (horizontal)
    d : float
        Distance between planes and the size of the pixel
    precision_scaling_factor : int
        Scaling factor for the image domain (for more accurate rt)
    idx : int
        The current pixel-index for which intersections will be computed

    Returns:
    int_f_xs, int_f_ys, int_b_xs, int_b_ys : array_like
        Arrays of front and back intersections for the antenna position
        (The unintersected pixels follow the same logic as in circular
        or elliptic approximations)
    """

    new_shape = [
        np.size(ant_xs),
        np.size(binary_mask, axis=0) // precision_scaling_factor,
        np.size(binary_mask, axis=1) // precision_scaling_factor,
    ]

    ant_idx, y_idx_no_scale, x_idx_no_scale = np.unravel_index(idx, new_shape)

    x_idx = x_idx_no_scale * precision_scaling_factor
    y_idx = y_idx_no_scale * precision_scaling_factor

    ant_pos_x = ant_xs[ant_idx]
    ant_pos_y = ant_ys[ant_idx]

    alpha_x = np.array([])
    alpha_y = np.array([])

    alpha_x_0 = (x_planes[0] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)
    alpha_x_n = (x_planes[-1] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)

    alpha_y_0 = (y_planes[0] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)
    alpha_y_n = (y_planes[-1] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)

    alpha_min = max(0.0, min(alpha_x_0, alpha_x_n), min(alpha_y_0, alpha_y_n))
    alpha_max = min(1.0, max(alpha_x_0, alpha_x_n), max(alpha_y_0, alpha_y_n))

    if pix_xs[x_idx] - ant_pos_x > 0:
        i_min = np.size(x_planes) - np.floor(
            (x_planes[-1] - alpha_min * (pix_xs[x_idx] - ant_pos_x) - ant_pos_x)
            / d
        ).astype("int")

        i_max = 1 + np.floor(
            (ant_pos_x + alpha_max * (pix_xs[x_idx] - ant_pos_x) - x_planes[0])
            / d
        ).astype("int")

        if i_min >= 0 and i_max >= 0:
            i_set = np.arange(i_min, i_max)
            alpha_x = (x_planes[i_set] - ant_pos_x) / (
                pix_xs[x_idx] - ant_pos_x
            )

    else:
        i_min = np.size(x_planes) - np.floor(
            (x_planes[-1] - alpha_max * (pix_xs[x_idx] - ant_pos_x) - ant_pos_x)
            / d
        ).astype("int")

        i_max = 1 + np.floor(
            (ant_pos_x + alpha_min * (pix_xs[x_idx] - ant_pos_x) - x_planes[0])
            / d
        ).astype("int")

        if i_min >= 0 and i_max >= 0:
            i_set = np.arange(i_min, i_max)
            i_set = np.flip(i_set)
            alpha_x = (x_planes[i_set] - ant_pos_x) / (
                pix_xs[x_idx] - ant_pos_x
            )

    if pix_ys[y_idx] - ant_pos_y > 0:
        j_min = np.size(y_planes) - np.floor(
            (y_planes[-1] - alpha_min * (pix_ys[y_idx] - ant_pos_y) - ant_pos_y)
            / d
        ).astype("int")

        j_max = 1 + np.floor(
            (ant_pos_y + alpha_max * (pix_ys[y_idx] - ant_pos_y) - y_planes[0])
            / d
        ).astype("int")

        if j_min >= 0 and j_max >= 0:
            j_set = np.arange(j_min, j_max)
            alpha_y = (y_planes[j_set] - ant_pos_y) / (
                pix_ys[y_idx] - ant_pos_y
            )

    else:
        j_min = np.size(y_planes) - np.floor(
            (y_planes[-1] - alpha_max * (pix_ys[y_idx] - ant_pos_y) - ant_pos_y)
            / d
        ).astype("int")

        j_max = 1 + np.floor(
            (ant_pos_y + alpha_min * (pix_ys[y_idx] - ant_pos_y) - y_planes[0])
            / d
        ).astype("int")

        if j_min >= 0 and j_max >= 0:
            j_set = np.arange(j_min, j_max)
            j_set = np.flip(j_set)
            alpha_y = (y_planes[j_set] - ant_pos_y) / (
                pix_ys[y_idx] - ant_pos_y
            )

    alpha_no_minmax = np.unique(np.concatenate((alpha_x, alpha_y)))

    if alpha_no_minmax.size != 0:
        alpha = np.unique(
            np.concatenate(
                (np.array([alpha_min]), alpha_no_minmax, np.array([alpha_max]))
            )
        )
    else:
        alpha = np.unique(np.array([alpha_min, alpha_max]))

    alpha = alpha[(alpha >= 0.0) & (alpha <= 1.0)]

    alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])

    intersected_x_idxs = np.floor(
        (ant_pos_x + alpha_mid * (pix_xs[x_idx] - ant_pos_x) - x_planes[0]) / d
    ).astype("int")

    intersected_y_idxs = np.floor(
        (ant_pos_y + alpha_mid * (pix_ys[y_idx] - ant_pos_y) - y_planes[0]) / d
    ).astype("int")

    crop_x = (intersected_x_idxs >= 0) & (
        intersected_x_idxs < np.size(binary_mask, axis=0)
    )
    intersected_x_idxs = intersected_x_idxs[crop_x]
    intersected_y_idxs = intersected_y_idxs[crop_x]

    crop_y = (intersected_y_idxs >= 0) & (
        intersected_y_idxs < np.size(binary_mask, axis=0)
    )
    intersected_x_idxs = intersected_x_idxs[crop_y]
    intersected_y_idxs = intersected_y_idxs[crop_y]

    intersected_y_idxs = np.size(binary_mask, axis=0) - 1 - intersected_y_idxs

    return np.array(
        intersections_per_ray(
            binary_mask,
            intersected_x_idxs,
            intersected_y_idxs,
            pix_xs,
            pix_ys,
            pix_xs[x_idx],
            pix_ys[y_idx],
        )
    )
