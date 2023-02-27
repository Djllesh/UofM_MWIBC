"""
Illia Prykhodko
University of Manitoba
January 24th, 2023
"""

from umbms.beamform.extras import get_ant_scan_xys
import numpy as np
from functools import partial


###############################################################################


def parallel_time_raytrace(pix_angs, pix_dists_from_center, speed_map, ant_xs,
                           ant_ys, ant_x_idxs, ant_y_idxs, pix_width,
                           first_plane_dist, possible_x_idxs, possible_y_idxs,
                           roi_rad, pix_dists_one_dimension, idx):
    """Get the time-of-flight from ant_pos to pix via parallelized
    ray-tracing

    Computes the propagation time-of-flight for the signal from one
    antenna position to one pixel in the image-space, and back. This
    function is designed to be used via parallel processing, with idx
    being from an iterable.

    Parameters
    ----------
    pix_angs : array_like
        The angle of each pixel off of the central-axis of the antenna,
        for each antenna position, in degrees
    pix_dists_from_center : array_like
        The polar distance from the center of the image-space to each
        pixel, for each antenna position, in meters
    speed_map : array_like
        Map of the estimated propagation speeds in the image-space,
        in m/s
    ant_xs : array_like
        The x-positions of each antenna position in the scan, in meters
    ant_ys : array_like
        The y-positions of each antenna position in the scan, in meters
    ant_x_idxs : array_like
        The x-coordinates of each antenna position in the scan
        (i.e., the x-indices of the pixels of each antenna position in
        the scan)
    ant_y_idxs : array_like
        The y-coordinates of each antenna position in the scan
        (i.e., the y-indices of the pixels of each antenna position in
        the scan)
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of the first plane used to segment the
        image-space, in meters
    possible_x_idxs : array_like
        The arr of the possible x-indices of the intersected
        (intersected by the ray-of-propagation) pixels
    possible_y_idxs : array_like
        The arr of the possible y-indices of the intersected
        (intersected by the ray-of-propagation) pixels
    roi_rad : float
        The radius of the central circular region-of-interest, in
        meters - the time-of-flight is only computed for pixels within
        this central region, to save computation time
    pix_dists_one_dimension : array_like
        1D arr of the pixel positions, in meters, along either dimension
        of the image-space
    idx : int
        The current pixel-index for which the propagation time will be
        computed

    Returns
    -------
    1, or the true time-of flight of the signal
    """

    # Find the current antenna position and x/y-indices of the target pixel
    ant_pos, x_idx, y_idx = np.unravel_index(idx, np.shape(pix_angs))

    # If this pixel is both in front of the antenna and within the central
    # region-of-interest
    if (np.abs(pix_angs[ant_pos, x_idx, y_idx]) < 90 and
            (pix_dists_from_center[x_idx, y_idx] < roi_rad)):

        # Find the x/y-positions of the origin of this ray (i.e., the
        # x/y-positions of the antenna here)
        ray_ini_x = ant_xs[ant_pos]
        ray_ini_y = ant_ys[ant_pos]

        # Return the time-of-flight to this pixel
        return get_tof(speed_map, ant_x_idxs[ant_pos],
                       ant_y_idxs[ant_pos], x_idx, y_idx, ray_ini_x,
                       ray_ini_y, pix_dists_one_dimension[x_idx],
                       pix_dists_one_dimension[y_idx], pix_width,
                       first_plane_dist, possible_x_idxs,
                       possible_y_idxs)

    # If the pixel is either behind the antenna OR not in the central
    # region-of-interest, return 1 (the time-of-response is set to 1
    # second - a very long time)
    else:
        return 1


def parallel_time_attn_raytrace(ref_coeffs, pix_angs, pix_dists_from_center,
                                speed_map, ant_xs, ant_ys, ant_x_idxs,
                                ant_y_idxs, pix_width, first_plane_dist,
                                possible_x_idxs, possible_y_idxs, roi_rad,
                                pix_dists_one_dimension, idx):
    """Get the time-of-flight and attenuation factor for a pixel

    Computes the propagation time-of-flight for the signal from one
    antenna position to one pixel in the image-space, and back, and the
    attenuation factor. This function is designed to be used via
    parallel processing, with idx being from an iterable.

    Parameters
    ----------
    ref_coeffs : array_like
        The reflection coefficients for each pixel in the image-space
    pix_angs : array_like
        The angle of each pixel off of the central-axis of the antenna,
        for each antenna position, in degrees
    pix_dists_from_center : array_like
        The polar distance from the center of the image-space to each
        pixel, for each antenna position, in meters
    speed_map : array_like
        Map of the estimated propagation speeds in the image-space,
        in m/s
    ant_xs : array_like
        The x-positions of each antenna position in the scan, in meters
    ant_ys : array_like
        The y-positions of each antenna position in the scan, in meters
    ant_x_idxs : array_like
        The x-coordinates of each antenna position in the scan
        (i.e., the x-indices of the pixels of each antenna position in
        the scan)
    ant_y_idxs : array_like
        The y-coordinates of each antenna position in the scan
        (i.e., the y-indices of the pixels of each antenna position in
        the scan)
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of the first plane used to segment the
        image-space, in meters
    possible_x_idxs : array_like
        The arr of the possible x-indices of the intersected
        (intersected by the ray-of-propagation) pixels
    possible_y_idxs : array_like
        The arr of the possible y-indices of the intersected
        (intersected by the ray-of-propagation) pixels
    roi_rad : float
        The radius of the central circular region-of-interest, in
        meters - the time-of-flight is only computed for pixels within
        this central region, to save computation time
    pix_dists_one_dimension : array_like
        1D arr of the pixel positions, in meters, along either dimension
        of the image-space
    idx : int
        The current pixel-index for which the propagation time will be
        computed

    Returns
    -------
    (1, 1), or (true time of flight, true attenuation factor)
    """

    # Find the current antenna position, and the x/y indices of the target
    # pixel
    ant_pos, x_idx, y_idx = np.unravel_index(idx, np.shape(pix_angs))

    # If the target pixel is in front of the antenna and wihtin the central
    # region of interest
    if np.abs(pix_angs[ant_pos, x_idx, y_idx]) < 90 and \
            (pix_dists_from_center[x_idx, y_idx] < roi_rad):

        # Find the x/y-position of the origin of the ray here (i.e., the
        # x/y positions of the antenna position here)
        ray_ini_x = ant_xs[ant_pos]
        ray_ini_y = ant_ys[ant_pos]

        # Return the time of flight and the attenuation factor
        return get_tof_attn(speed_map, ref_coeffs, ant_x_idxs[ant_pos],
                            ant_y_idxs[ant_pos], x_idx, y_idx, ray_ini_x,
                            ray_ini_y, pix_dists_one_dimension[x_idx],
                            pix_dists_one_dimension[y_idx], pix_width,
                            first_plane_dist, possible_x_idxs,
                            possible_y_idxs)
    else:
        return 1, 1


def get_tof(speed_map, ini_x_idx, ini_y_idx, fin_x_idx, fin_y_idx, ini_x,
            ini_y, fin_x, fin_y, pix_width, first_plane_dist, possible_x_idxs,
            possible_y_idxs):
    """Ray-trace to obtain the time-of-flight (tof) of the return signal

    Uses the Siddon ray-tracing algorithm to compute the time-of-flight
    of the microwave signal from the starting pixel to the end pixel.

    Parameters
    ----------
    speed_map : array_like
        Map of the estimated propagation speeds of the microwave signal
        in the image-space, in m/s
    ini_x_idx : int
        The x-index of the pixel from which the ray originates
    ini_y_idx : int
        The y-index of the pixel from which the ray originates
    fin_x_idx : int
        The x-index of the pixel from which the ray terminates
        (i.e., the target pixel)
    fin_y_idx : int
        The y-index of the pixel from which the ray terminates
        (i.e., the target pixel)
    ini_x : float
        The physical x-position of the pixel from which the ray
        originates, in meters
    ini_y : float
        The physical y-position of the pixel from which the ray
         originates, in meters
    fin_x : float
        The physical x-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    fin_y : float
        The physical y-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of zeroth plane used to define the
        pixel-grid, in meters
    possible_x_idxs : array_like
        Array of the possible x-indices of the intersected pixels
    possible_y_idxs : array_like
        Array of the possible y-indices of the intersected pixels

    Returns
    -------
    tof : float
        The time-of-flight of the signal from the start position to the
        end position and back, in seconds
    """

    # Find the difference in the x/y positions of the start and end positions
    x_diff = fin_x - ini_x
    y_diff = - fin_y - ini_y

    # If the start/end positions are the same, return 1 (a large
    # time-of-response)
    if x_diff == 0 and y_diff == 0:
        tof = 1

    # If the start/end x-positions are the same, but the start/end y-positions
    # are different
    elif x_diff == 0 and y_diff != 0:

        # If the end y-position is greater than the start y-position
        if y_diff > 0:

            # Get the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(ini_y_idx, fin_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

        else:  # If the start y-position is greater than the end y-position

            # Get the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(fin_y_idx, ini_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

    # If the start/end x-positions are different, but the start/end
    # y-positions are the same
    elif x_diff != 0 and y_diff == 0:

        # Find the distance the ray travels one-way
        ray_length = np.abs(x_diff)

        # If the end x-position is greater than the start x-position
        if x_diff > 0:

            # Get the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(ini_x_idx, fin_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the physical width step-size for this ray
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

        else:  # If the start x-position is greater than the end x-position

            # Get the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(fin_x_idx, ini_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the physical width step-size for this ray
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

    # If the start/end x-positions and y-positions are different
    else:

        # Find the alpha values along the x/y dimensions for the
        # intersections, as in the Siddon algorithm
        alpha_xs = (first_plane_dist + possible_x_idxs * pix_width
                    - ini_x) / x_diff
        alpha_ys = (first_plane_dist + possible_y_idxs * pix_width
                    - ini_y) / y_diff

        # Retain values between 0 and 1, corresponding to true pixel
        # intersection
        alpha_xs = alpha_xs[(alpha_xs > 0) & (alpha_xs < 1)]
        alpha_ys = alpha_ys[(alpha_ys > 0) & (alpha_ys < 1)]

        # Remove any non-unique values
        alpha_xys = np.unique(np.concatenate([alpha_xs, alpha_ys]))

        # Find the distance the ray propagates one-way
        ray_length = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # Find the length of each intersection
        pixel_intersection_lengths = ray_length * (alpha_xys[1:]
                                                   - alpha_xys[:-1])

        # Find the alpha values at the middle of each pixel
        mid_pixel_alphas = 0.5 * (alpha_xys[1:] + alpha_xys[:-1])

        # Find the x/y-indices of each intersected pixel
        intersected_x_idxs = np.floor((ini_x + mid_pixel_alphas * x_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1
        intersected_y_idxs = np.floor((ini_y + mid_pixel_alphas * y_diff -
                                       first_plane_dist)
                                      / pix_width).astype('int') - 1

        # Sum the propagation times through each pixel
        tof = np.sum(pixel_intersection_lengths /
                     speed_map[intersected_x_idxs, intersected_y_idxs])

    # Multiply the propagation time by 2 to account for propagation to the
    # target and back to the antenna
    tof *= 2

    return tof


def get_tof_attn(speed_map, ref_coeffs, ini_x_idx, ini_y_idx, fin_x_idx,
                 fin_y_idx, ini_x, ini_y, fin_x, fin_y, pix_width,
                 first_plane_dist, possible_x_idxs, possible_y_idxs):
    """Ray-trace to obtain the time-of-flight (tof) and attenuation
    factor

    Uses the Siddon ray-tracing algorithm to compute the time-of-flight
    and attenuation factor of the microwave signal from the starting
    pixel to the end pixel.

    Parameters
    ----------
    speed_map : array_like
        Map of the estimated propagation speeds of the microwave signal
        in the image-space, in m/s
    ref_coeffs : array_like
        Map of the reflection coefficients in the image-space
    ini_x_idx : int
        The x-index of the pixel from which the ray originates
    ini_y_idx : int
        The y-index of the pixel from which the ray originates
    fin_x_idx : int
        The x-index of the pixel from which the ray terminates (i.e.,
        the target pixel)
    fin_y_idx : int
        The y-index of the pixel from which the ray terminates (i.e.,
        the target pixel)
    ini_x : float
        The physical x-position of the pixel from which the ray
        originates, in meters
    ini_y : float
        The physical y-position of the pixel from which the ray
        originates, in meters
    fin_x : float
        The physical x-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    fin_y : float
        The physical y-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of zeroth plane used to define the
        pixel-grid, in meters
    possible_x_idxs : array_like
        Array of the possible x-indices of the intersected pixels
    possible_y_idxs : array_like
        Array of the possible y-indices of the intersected pixels

    Returns
    -------
    tof : float
        The time-of-flight of the signal from the start position to
        the end position and back, in seconds
    attn : float
        The attenuation factor for the signal from the start position to
        the end position
    """

    # Find the differences in the x/y-positions of the start and end positions
    x_diff = fin_x - ini_x
    y_diff = fin_y - ini_y

    # If the start/end positions are the same
    if x_diff == 0 and y_diff == 0:

        # Return a large value for the propagation time, and return a value
        # indicating no signal attenuation
        tof = 1
        attn = 0

    # If the start/end positions have the same x-position but different
    # y-positions
    elif x_diff == 0 and y_diff != 0:

        # Find the length the ray propagates one-way
        ray_length = np.abs(y_diff)

        # If the y-position of the end position is greater the start position
        if y_diff > 0:

            # Find the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(ini_y_idx, fin_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_y_idxs) + 1)

            # Sum the propagation times from each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs]) ** 2)

        # If the y-position of the end position is lesser than the start
        # position
        else:

            # Find the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(fin_y_idx, ini_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_y_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs]) ** 2)

    # If the x-positions of the start/end positions are the same, but the
    # y-positions are different
    elif x_diff != 0 and y_diff == 0:

        # Find the length the ray propagates one-way
        ray_length = np.abs(x_diff)

        # If the x-position of the end position is greater than the start
        # position
        if x_diff > 0:

            # Find the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(ini_x_idx, fin_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs]) ** 2)

        # If the x-position of the end position is lesser than the start
        # position
        else:

            # Find the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(fin_x_idx, ini_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs]) ** 2)

    # If the x-positions and the y-positions of the start and end pixels
    # are both different
    else:

        # Find the alpha values along the x/y dimensions for the
        # intersections, as in the Siddon algorithm
        alpha_xs = (first_plane_dist + possible_x_idxs * pix_width
                    - ini_x) / x_diff
        alpha_ys = (first_plane_dist + possible_y_idxs * pix_width
                    - ini_y) / y_diff

        # Retain values between 0 and 1, corresponding to true
        # pixel intersection
        alpha_xs = alpha_xs[(alpha_xs > 0) & (alpha_xs < 1)]
        alpha_ys = alpha_ys[(alpha_ys > 0) & (alpha_ys < 1)]

        # Remove any non-unique values
        alpha_xys = np.unique(np.concatenate([alpha_xs, alpha_ys]))

        # Find the distance the ray propagates one-way
        ray_length = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # Find the intersection lengths of the ray intersections in each pixel
        intersection_lengths = ray_length * (alpha_xys[1:] - alpha_xys[:-1])

        # Find the alpha values at the middle of each pixel
        mid_pixel_alphas = 0.5 * (alpha_xys[1:] + alpha_xys[:-1])

        # Find the x/y-indices of the intersected pixels
        intersected_x_idxs = np.floor((ini_x + mid_pixel_alphas * x_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1
        intersected_y_idxs = np.floor((ini_y + mid_pixel_alphas * y_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1

        # Sum the propagation times for the ray through each pixel
        tof = np.sum(intersection_lengths
                     / speed_map[intersected_x_idxs, intersected_y_idxs])

        # Multiply the attenuation factors through each pixel
        attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                       intersected_y_idxs]) ** 2)

    # Multiply the propagation time by two to account for propagation
    # to/from the target pixel
    tof *= 2

    return tof, attn


def find_boundary_rt(binary_mask, ant_rad, roi_rad, n_ant_pos=72,
                     ini_ant_ang=-136.0, worker_pool=None):
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
    m_size = np.size(binary_mask, axis=0)

    # initializing arrays for storing intersection coordinates
    # front intersection - closer to antenna
    int_f_xs = np.zeros([n_ant_pos, m_size, m_size], dtype=float)
    int_f_ys = np.zeros_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.zeros_like(int_f_xs)
    int_b_ys = np.zeros_like(int_f_xs)

    # obtain antenna and pixel coordinates
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad=ant_rad,
                                      n_ant_pos=n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)
    pix_xs = np.linspace(-roi_rad, roi_rad, m_size)
    pix_ys = np.flip(pix_xs)
    d = pix_xs[1] - pix_xs[0]

    # initialize planes
    x_planes = np.linspace(-roi_rad, -roi_rad + d * m_size, m_size + 1)
    y_planes = x_planes

    if worker_pool is not None:

        iterable_idx = range(np.size(binary_mask) * n_ant_pos)

        parallel_func = partial(parallel_intersections_per_ant_pos_rt,
                                binary_mask, ant_xs, ant_ys, pix_xs, pix_ys,
                                x_planes, y_planes, d)

        intersections = np.array(worker_pool.map(parallel_func, iterable_idx))

        int_f_xs[:, :, :] = np.reshape(intersections[:, 0],
                                       [np.size(ant_xs),
                                        np.size(binary_mask, axis=0),
                                        np.size(binary_mask, axis=1)])
        int_f_ys[:, :, :] = np.reshape(intersections[:, 1],
                                       [np.size(ant_xs),
                                        np.size(binary_mask, axis=0),
                                        np.size(binary_mask, axis=1)])
        int_b_xs[:, :, :] = np.reshape(intersections[:, 2],
                                       [np.size(ant_xs),
                                        np.size(binary_mask, axis=0),
                                        np.size(binary_mask, axis=1)])
        int_b_ys[:, :, :] = np.reshape(intersections[:, 3],
                                       [np.size(ant_xs),
                                        np.size(binary_mask, axis=0),
                                        np.size(binary_mask, axis=1)])

    else:

        for ant_pos in range(n_ant_pos):
            int_f_xs[ant_pos, :, :], int_f_ys[ant_pos, :, :], \
            int_b_xs[ant_pos, :, :], int_b_ys[ant_pos, :, :] = \
                intersections_per_antenna_pos_rt(binary_mask, ant_xs[ant_pos],
                                                 ant_ys[ant_pos], pix_xs,
                                                 pix_ys, x_planes, y_planes, d)

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def intersections_per_antenna_pos_rt(binary_mask, ant_pos_x, ant_pos_y,
                                     pix_xs, pix_ys, x_planes, y_planes, d):
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

    Returns:
    int_f_xs, int_f_ys, int_b_xs, int_b_ys : array_like
        Arrays of front and back intersections for the antenna position
        (The unintersected pixels follow the same logic as in circular
        or elliptic approximations)
    """

    int_f_xs = np.zeros([np.size(binary_mask, axis=0),
                         np.size(binary_mask, axis=0)], dtype=float)
    int_f_ys = np.zeros_like(int_f_xs)
    int_b_xs = np.zeros_like(int_f_xs)
    int_b_ys = np.zeros_like(int_f_xs)

    for x_idx in range(np.size(binary_mask, axis=0)):
        for y_idx in range(np.size(binary_mask, axis=0)):

            alpha_x = np.array([])
            alpha_y = np.array([])

            alpha_x_0 = (x_planes[0] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)
            alpha_x_n = (x_planes[-1] - ant_pos_x) / (
                    pix_xs[x_idx] - ant_pos_x)

            alpha_y_0 = (y_planes[0] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)
            alpha_y_n = (y_planes[-1] - ant_pos_y) / (
                    pix_ys[y_idx] - ant_pos_y)

            alpha_min = max(0., min(alpha_x_0, alpha_x_n),
                            min(alpha_y_0, alpha_y_n))

            alpha_max = min(1., max(alpha_x_0, alpha_x_n),
                            max(alpha_y_0, alpha_y_n))

            if pix_xs[x_idx] - ant_pos_x > 0:

                i_min = np.size(x_planes) - \
                        np.floor((x_planes[-1] - alpha_min *
                                  (pix_xs[x_idx] - ant_pos_x)
                                  - ant_pos_x) / d).astype('int')

                i_max = 1 + np.floor((ant_pos_x + alpha_max *
                                      (pix_xs[x_idx] - ant_pos_x)
                                      - x_planes[0]) / d).astype('int')

                if i_min >= 0 and i_max >= 0:
                    i_set = np.arange(i_min, i_max)
                    alpha_x = (x_planes[i_set] - ant_pos_x) / (pix_xs[x_idx]
                                                               - ant_pos_x)

            else:

                i_min = np.size(x_planes) - \
                        np.floor((x_planes[-1] - alpha_max *
                                  (pix_xs[x_idx] - ant_pos_x)
                                  - ant_pos_x) / d).astype('int')

                i_max = 1 + np.floor((ant_pos_x + alpha_min *
                                      (pix_xs[x_idx] - ant_pos_x)
                                      - x_planes[0]) / d).astype('int')

                if i_min >= 0 and i_max >= 0:
                    i_set = np.arange(i_min, i_max)
                    i_set = np.flip(i_set)
                    alpha_x = (x_planes[i_set] - ant_pos_x) \
                              / (pix_xs[x_idx] - ant_pos_x)

            if pix_ys[y_idx] - ant_pos_y > 0:

                j_min = np.size(y_planes) - \
                        np.floor((y_planes[-1] - alpha_min *
                                  (pix_ys[y_idx] - ant_pos_y)
                                  - ant_pos_y) / d).astype('int')

                j_max = 1 + np.floor((ant_pos_y +
                                      alpha_max * (pix_ys[y_idx] - ant_pos_y)
                                      - y_planes[0]) / d).astype('int')

                if j_min >= 0 and j_max >= 0:
                    j_set = np.arange(j_min, j_max)
                    alpha_y = (y_planes[j_set] - ant_pos_y) \
                              / (pix_ys[y_idx] - ant_pos_y)

            else:

                j_min = np.size(y_planes) - \
                        np.floor((y_planes[-1] - alpha_max *
                                  (pix_ys[y_idx] - ant_pos_y)
                                  - ant_pos_y) / d).astype('int')

                j_max = 1 + np.floor((ant_pos_y +
                                      alpha_min * (pix_ys[y_idx] - ant_pos_y)
                                      - y_planes[0]) / d).astype('int')

                if j_min >= 0 and j_max >= 0:
                    j_set = np.arange(j_min, j_max)
                    j_set = np.flip(j_set)
                    alpha_y = (y_planes[j_set] - ant_pos_y) \
                              / (pix_ys[y_idx] - ant_pos_y)

            alpha_no_minmax = np.unique(np.concatenate((alpha_x, alpha_y)))

            if alpha_no_minmax.size != 0:
                alpha = np.unique(np.concatenate((np.array([alpha_min]),
                                                  alpha_no_minmax,
                                                  np.array([alpha_max]))))
            else:
                alpha = np.unique(np.array([alpha_min, alpha_max]))

            alpha = alpha[(alpha >= 0.) & (alpha <= 1.)]

            alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])

            intersected_x_idxs = np.floor((ant_pos_x + alpha_mid *
                                           (pix_xs[x_idx] - ant_pos_x)
                                           - x_planes[0]) / d).astype('int')

            intersected_y_idxs = np.floor((ant_pos_y + alpha_mid *
                                           (pix_ys[y_idx] - ant_pos_y)
                                           - y_planes[0]) / d).astype('int')

            crop_x = (intersected_x_idxs >= 0) & \
                     (intersected_x_idxs < np.size(binary_mask, axis=0))
            intersected_x_idxs = intersected_x_idxs[crop_x]
            intersected_y_idxs = intersected_y_idxs[crop_x]

            crop_y = (intersected_y_idxs >= 0) & \
                     (intersected_y_idxs < np.size(binary_mask, axis=0))
            intersected_x_idxs = intersected_x_idxs[crop_y]
            intersected_y_idxs = intersected_y_idxs[crop_y]

            intersected_y_idxs = 149 - intersected_y_idxs

            int_f_xs[y_idx, x_idx], int_f_ys[y_idx, x_idx], \
            int_b_xs[y_idx, x_idx], int_b_ys[y_idx, x_idx] = \
                intersections_per_ray(binary_mask, intersected_x_idxs,
                                      intersected_y_idxs, pix_xs, pix_ys,
                                      pix_xs[x_idx], pix_ys[y_idx])

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def intersections_per_ray(binary_mask, intersected_x_idxs, intersected_y_idxs,
                          pix_xs, pix_ys, pix_x, pix_y):
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

    if intersected_x_idxs.size == 0:
        int_f_x = pix_x
        int_f_y = pix_y
        int_b_x = pix_x
        int_b_y = pix_y

        return int_f_x, int_f_y, int_b_x, int_b_y

    if binary_mask[intersected_y_idxs[-1], intersected_x_idxs[-1]]:
        int_b_x = pix_x
        int_b_y = pix_y

    binary_ray = binary_mask[intersected_y_idxs, intersected_x_idxs]
    inside_idxs = np.argwhere(binary_ray).flatten()

    if inside_idxs.size != 0:

        int_f_x = (pix_xs[intersected_x_idxs[inside_idxs[0]] - 1] +
                   pix_xs[intersected_x_idxs[inside_idxs[0]] + 1]) / 2
        int_f_y = (pix_ys[intersected_y_idxs[inside_idxs[0]] - 1] +
                   pix_ys[intersected_y_idxs[inside_idxs[0]] + 1]) / 2

        if np.isnan(int_b_x):
            int_b_x = (pix_xs[intersected_x_idxs[inside_idxs[-1]] - 1] +
                       pix_xs[intersected_x_idxs[inside_idxs[-1]] + 1]) / 2

            int_b_y = (pix_ys[intersected_y_idxs[inside_idxs[-1]] - 1] +
                       pix_ys[intersected_y_idxs[inside_idxs[-1]] + 1]) / 2

    else:

        int_f_x = pix_x
        int_f_y = pix_y
        int_b_x = pix_x
        int_b_y = pix_y

    return int_f_x, int_f_y, int_b_x, int_b_y


def parallel_intersections_per_ant_pos_rt(binary_mask, ant_xs, ant_ys,
                                          pix_xs, pix_ys, x_planes, y_planes,
                                          d, idx):
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
    idx : int
        The current pixel-index for which intersections will be computed

    Returns:
    int_f_xs, int_f_ys, int_b_xs, int_b_ys : array_like
        Arrays of front and back intersections for the antenna position
        (The unintersected pixels follow the same logic as in circular
        or elliptic approximations)
    """

    new_shape = [np.size(ant_xs),
                 np.size(binary_mask, axis=0),
                 np.size(binary_mask, axis=1)]

    ant_idx, y_idx, x_idx = np.unravel_index(idx, new_shape)

    ant_pos_x = ant_xs[ant_idx]
    ant_pos_y = ant_ys[ant_idx]

    alpha_x = np.array([])
    alpha_y = np.array([])

    alpha_x_0 = (x_planes[0] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)
    alpha_x_n = (x_planes[-1] - ant_pos_x) / (pix_xs[x_idx] - ant_pos_x)

    alpha_y_0 = (y_planes[0] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)
    alpha_y_n = (y_planes[-1] - ant_pos_y) / (pix_ys[y_idx] - ant_pos_y)

    alpha_min = max(0., min(alpha_x_0, alpha_x_n), min(alpha_y_0, alpha_y_n))
    alpha_max = min(1., max(alpha_x_0, alpha_x_n), max(alpha_y_0, alpha_y_n))

    if pix_xs[x_idx] - ant_pos_x > 0:

        i_min = np.size(x_planes) - \
                np.floor((x_planes[-1] - alpha_min *
                          (pix_xs[x_idx] - ant_pos_x)
                          - ant_pos_x) / d).astype('int')

        i_max = 1 + np.floor((ant_pos_x + alpha_max *
                              (pix_xs[x_idx] - ant_pos_x)
                              - x_planes[0]) / d).astype('int')

        if i_min >= 0 and i_max >= 0:
            i_set = np.arange(i_min, i_max)
            alpha_x = (x_planes[i_set] - ant_pos_x) \
                      / (pix_xs[x_idx] - ant_pos_x)

    else:

        i_min = np.size(x_planes) - \
                np.floor((x_planes[-1] - alpha_max *
                          (pix_xs[x_idx] - ant_pos_x)
                          - ant_pos_x) / d).astype('int')

        i_max = 1 + np.floor((ant_pos_x + alpha_min *
                              (pix_xs[x_idx] - ant_pos_x)
                              - x_planes[0]) / d).astype('int')

        if i_min >= 0 and i_max >= 0:
            i_set = np.arange(i_min, i_max)
            i_set = np.flip(i_set)
            alpha_x = (x_planes[i_set] - ant_pos_x) \
                      / (pix_xs[x_idx] - ant_pos_x)

    if pix_ys[y_idx] - ant_pos_y > 0:

        j_min = np.size(y_planes) - \
                np.floor((y_planes[-1] - alpha_min
                          * (pix_ys[y_idx] - ant_pos_y)
                          - ant_pos_y) / d).astype('int')

        j_max = 1 + np.floor((ant_pos_y +
                              alpha_max * (pix_ys[y_idx] - ant_pos_y)
                              - y_planes[0]) / d).astype('int')

        if j_min >= 0 and j_max >= 0:
            j_set = np.arange(j_min, j_max)
            alpha_y = (y_planes[j_set] - ant_pos_y) \
                      / (pix_ys[y_idx] - ant_pos_y)

    else:

        j_min = np.size(y_planes) - \
                np.floor((y_planes[-1] - alpha_max
                          * (pix_ys[y_idx] - ant_pos_y)
                          - ant_pos_y) / d).astype('int')

        j_max = 1 + np.floor((ant_pos_y +
                              alpha_min * (pix_ys[y_idx] - ant_pos_y)
                              - y_planes[0]) / d).astype('int')

        if j_min >= 0 and j_max >= 0:
            j_set = np.arange(j_min, j_max)
            j_set = np.flip(j_set)
            alpha_y = (y_planes[j_set] - ant_pos_y) \
                      / (pix_ys[y_idx] - ant_pos_y)

    alpha_no_minmax = np.unique(np.concatenate((alpha_x, alpha_y)))

    if alpha_no_minmax.size != 0:
        alpha = np.unique(np.concatenate((np.array([alpha_min]),
                                          alpha_no_minmax,
                                          np.array([alpha_max]))))
    else:
        alpha = np.unique(np.array([alpha_min, alpha_max]))

    alpha = alpha[(alpha >= 0.) & (alpha <= 1.)]

    alpha_mid = 0.5 * (alpha[1:] + alpha[:-1])

    intersected_x_idxs = np.floor((ant_pos_x + alpha_mid *
                                   (pix_xs[x_idx] - ant_pos_x)
                                   - x_planes[0]) / d).astype('int')

    intersected_y_idxs = np.floor((ant_pos_y + alpha_mid *
                                   (pix_ys[y_idx] - ant_pos_y)
                                   - y_planes[0]) / d).astype('int')

    crop_x = (intersected_x_idxs >= 0) \
             & (intersected_x_idxs < np.size(binary_mask, axis=0))
    intersected_x_idxs = intersected_x_idxs[crop_x]
    intersected_y_idxs = intersected_y_idxs[crop_x]

    crop_y = (intersected_y_idxs >= 0) \
             & (intersected_y_idxs < np.size(binary_mask, axis=0))
    intersected_x_idxs = intersected_x_idxs[crop_y]
    intersected_y_idxs = intersected_y_idxs[crop_y]

    intersected_y_idxs = 149 - intersected_y_idxs

    return np.array(intersections_per_ray(binary_mask, intersected_x_idxs,
                                          intersected_y_idxs, pix_xs, pix_ys,
                                          pix_xs[x_idx], pix_ys[y_idx]))
