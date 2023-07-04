"""
Prykhodko Illia
University of Manitoba
November 17, 2022
"""
import os
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.constants import speed_of_light
from umbms.beamform.iczt import iczt
from umbms.beamform.utility import get_xy_arrs
import matplotlib.pyplot as plt

__VAC_SPEED = speed_of_light


def rot(theta):
    """Returns a 2-D rotation matrix

    Parameters:
    ------------
    theta : float
        Radian angle of rotation (counter clockwise)

    Returns:
    ------------
    rot_matr : array_like 2x2
        Rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def find_boundary(img, roi_rad, n_slices=50, threshold=0.005):
    """Finds the approximate outline of a breast phantom.
    Search is based on a highest intensity peaks that are the closest to
    the antenna

    Parameters
    ----------
    img : array_like, NxN
        Reconstructed reflectivity profile, breast phantom image.
        Works best if the reconstruction was performed in reference to
        an empty chamber scan
    roi_rad : float
        Radius of region of interest of the scan
    n_slices : int
        Number of slices that will be taken of the image
    threshold : float
        Precision value
    Returns
    ----------
    bound_x, bound_y : array_like n_slices x 1
        x and y coordinate arrays of breast boundary
    """
    # unit rotation
    theta = 360 / n_slices

    # initialize arrays
    bound_x, bound_y = np.array([]), np.array([])
    # one pixel distance in meters
    dx = 2 * roi_rad / np.size(img, axis=0)

    for i in range(n_slices):
        # create a temporary rotated image
        img_to_slice = ndimage.rotate(img, i * theta,
                                      reshape=False, prefilter=False)

        # take a slice right in the middle that corresponds to y = 0
        slice = img_to_slice[np.size(img, axis=0) // 2, :]
        # normalize the slice
        max_val = np.amax(slice[~np.isnan(slice)])
        slice /= max_val

        # find two peaks
        peaks, _ = find_peaks(slice, height=0.4)

        # find corresponding x coordinates
        x = (peaks[0] - 74) * dx

        # temporary matrix of coordinates with no rotation
        to_rot = np.array([[x], [0]])

        # rotate back
        to_store = np.matmul(rot(np.deg2rad(-theta * i)), to_rot)

        if i != 0:
            distance = np.sqrt((to_store[0, 0] - bound_x[i - 1]) ** 2 +
                               (to_store[1, 0] - bound_y[i - 1]) ** 2)

            if distance > threshold:
                prev_peak = np.array([[bound_x[i - 1]], [bound_y[i - 1]]])
                new_idx = find_peak_threshold(slice, -i * theta, dx, prev_peak,
                                              threshold=threshold)

                x = (new_idx - 74) * dx
                to_store = np.matmul(rot(np.deg2rad(-i * theta)), [[x], [0]])

        bound_x = np.append(bound_x, to_store[0])
        bound_y = np.append(bound_y, to_store[1])

    return bound_x, bound_y


def find_peak_threshold(slice, theta, dx, prev_peak, threshold):
    """Finds a peak distance from which to previous isn't surpassing
    given threshold

    Parameters:
    ----------
    slice : array_like m_size x 1
        One-dimensional signal of reconstructed image
        along x-axis
    theta : float
        Rotation angle of a slice
    dx : float
        Transformation factor, one pix distance in meters
    prev_peak : array_like 2 x 1
        Coordinates of a previous peak in space domain

    threshold : float
        Maximal acceptable deviation

    Returns:
    ----------
    idx : int
        Index of a new peak
    """
    # decrease the height to find undetected peaks

    # get idx array to iterate through
    peaks, _ = find_peaks(slice)
    # for a case that several peaks are chosen
    under_threshold = np.array([], dtype=int)

    for peak_idx in peaks:  # for each index

        # find x_coordinate in spacial units
        x = (peak_idx - 74) * dx
        # find coordinates of the current peak with rotation
        peak_xy = np.matmul(rot(np.deg2rad(theta)), [[x], [0]])

        current_distance = np.sqrt((peak_xy[0, 0] - prev_peak[0, 0]) ** 2 +
                                   (peak_xy[1, 0] - prev_peak[1, 0]) ** 2)

        # append to a list of peaks that satisfy threshold
        if current_distance < threshold:
            under_threshold = np.append(under_threshold, peak_idx)

    # if only 1 found - return
    if np.size(under_threshold) == 1:
        return under_threshold[0]

    # if not, find a peak with minimal distance
    min_dist = 100
    idx_to_return = 0
    # iterate over all found indices
    for idx in under_threshold:
        peak_xy = np.matmul(rot(np.deg2rad(theta)), [[slice[idx]], [0]])
        current_distance = np.sqrt((peak_xy[0, 0] - prev_peak[0, 0]) ** 2 +
                                   (peak_xy[1, 0] - prev_peak[1, 0]) ** 2)

        if current_distance < min_dist:  # update minimal distance index
            min_dist = current_distance
            idx_to_return = idx

    return idx_to_return


def cart_to_polar(x_coords, y_coords):
    """Transforms cartesian coordinates to polar.
    Values are sorted wrt phi

    Parameters:
    ----------
    x_coords : array_like
    y_coords : array_like

    Return:
    rho, phi : array_like
    ----------
    """

    rho = np.sqrt(x_coords ** 2 + y_coords ** 2)
    phi = np.arctan2(y_coords, x_coords)
    negative = phi < 0
    phi[negative] = 2 * np.pi + phi[negative]
    rho, phi = sort_polar_angle(rho, phi)

    return rho, phi


def sort_polar_angle(rho, phi):
    """Returns two lists but sorted wrt angle

    Parameters:
    ----------
    rho : array_like
        Array of rho values
    phi : array_like
        Array of phi values

    Returns:
    ----------
    rho, phi : array_like
        Sorted arrays
    """

    list_of_tuples = []
    for (rh, ph) in zip(rho, phi):
        list_of_tuples.append((rh, ph))

    dtype = [('rho', float), ('phi', float)]
    ndarray_of_tuples = np.array(list_of_tuples, dtype=dtype)

    ndarray_of_tuples = np.sort(ndarray_of_tuples, order='phi')

    for i in range(np.size(rho)):
        rho[i] = ndarray_of_tuples[i][0]
        phi[i] = ndarray_of_tuples[i][1]

    return rho, phi


def polar_fit_cs(rho, phi):
    phi_ext = phi + np.pi * 2
    phi = np.concatenate((phi, phi_ext))
    rho = np.concatenate((rho, rho))
    return CubicSpline(phi, rho)


def get_binary_mask(cs, m_size, roi_rad, precision_scaling_factor=1):
    """Returns a binary mask that corresponds to breast phantom boundary

    Parameters:
    ----------
    cs : PPoly
        Cubic spline interpolation of a boundary
    m_size : int
        Size of the square image domain in pixels
    roi_rad : float
        Radius of ROI in meters
    precision_scaling_factor : int
        Scaling factor for the image domain (for more accurate rt)

    Returns:
    ----------
    mask : array_like
        Binary mask
    """
    pix_xs, pix_ys = get_xy_arrs(m_size=m_size * precision_scaling_factor,
                                 roi_rad=roi_rad)
    rho = np.sqrt(pix_xs ** 2 + pix_ys ** 2)
    phi = np.arctan2(pix_ys, pix_xs)
    negative = phi < 0
    phi[negative] = 2 * np.pi + phi[negative]
    rho_fit = cs(phi)
    mask = rho <= rho_fit
    return mask


def make_speed_map(mask, v_in, v_out=__VAC_SPEED):
    """Makes a binary speed map from a given shape mask

    Parameters:
    ------------
    mask : array_like
        Binary mask that determines the shape of the phanto
    v_in : float
        Propagation speed inside the boundary
    v_out : float
        Propagation speed outside the boundary
        (by default - speed of light)

    Returns:
    ------------
    speed_map : array_like
        Speed map for ray-tracing
    """

    m_size = np.size(mask, axis=0)
    speed_map = v_out * np.ones([m_size, m_size])
    speed_map[mask] = v_in

    return speed_map


def find_centre_of_mass(rho, phi):
    """Finds x,y coordinates of the centre of mass of a uniform 2d shape

    Parameters:
    ------------
    rho : array_like
        Array of polar radii
    phi : array_like
        Array of polar angles

    Returns:
    -----------
    x_cm, y_cm : float
        Cartesian coordinates of a centre of mass
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    x_cm = np.sum(x) / np.size(x)
    y_cm = np.sum(y) / np.size(y)

    return x_cm, y_cm


def find_centre_of_mass_from_cs(cs, n_points=200):
    """Finds x,y coordinates of the centre of mass of a uniform 2d shape
    approximated using CubicSpline

    Parameters:
    ------------
    cs : PPoly
        Cubic spline interpolation of a boundary
    n_points : int
        Number of points to create an array

    Returns:
    -----------
    x_cm, y_cm : float
        Cartesian coordinates of a centre of mass
    """

    phi = np.linspace(0, 2 * np.pi, 200)
    rho = cs(phi)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    x_cm = np.sum(x) / np.size(x)
    y_cm = np.sum(y) / np.size(y)

    return x_cm, y_cm


def get_boundary_iczt(adi_emp_cropped, ant_rad, n_ant_pos=72,
                      ini_ant_ang=-136.0, ini_t=0.5e-9, fin_t=5.5e-9,
                      n_time_pts=700, ini_f=2e9, fin_f=9e9, peak_threshold=10,
                      out_dir=''):

    """ Returns a CubicSpline interpolation
    function that approximates phantom boundary from a time-domain
    converted data

    Parameters:
    -----------
    adi_emp_cropped : array_like n_freqs X n_ant_pos
        Frequency-domain data of a scan minus empty chamber reference
    ant_rad : float
        Radius of antenna position AFTER all the corrections
    n_ant_pos : int
        Number of antenna positions during one scan
    ini_ant_ang : float
        Initial polar angle of an antenna, in degrees
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz
    peak_threshold : int
        The maximal allowed index jump in time points array
        (recommended not to change)
    Returns:
    ----------
    cs : PPoly
        Cubic spline interpolation of a boundary
    x_cm, y_cm : float
        Cartesian coordinates of a centre of mass
    """
    # convert frequency-domain data to time-domain
    td = iczt(adi_emp_cropped, ini_t=ini_t, fin_t=fin_t, n_time_pts=n_time_pts,
              ini_f=ini_f, fin_f=fin_f)

    # time response data
    ts = np.linspace(ini_t * 1e9, fin_t * 1e9, 700)

    # polar angle data
    angles = np.linspace(0, np.deg2rad(355), n_ant_pos) \
             + np.deg2rad(ini_ant_ang)

    # # angles for plotting
    plt_angles = np.linspace(0, 355, n_ant_pos)
    ts_plt = ts[:700]
    # #
    # # initializing an array of time-responces
    tr_threshold = np.array([])
    # tr_approx = np.array([])

    # creating an array of polar distances for storing
    rho = np.array([])

    # average signal for correlation
    avg_signal = np.abs(np.average(td, axis=1))

    time_aligned_signals = np.zeros_like(td, dtype=float)
    time_aligned_signals[:, 0] = np.abs(td[:, 0])
    first_peak, _ = find_peaks(time_aligned_signals[:, 0],
                               height=np.max(
                                   time_aligned_signals[:, 0]) - 1e-9)

    for ant_pos in range(1, np.size(td, axis=1), 1):
        signal_peak, _ = find_peaks(np.abs(td[:, ant_pos]),
                                    np.max(np.abs(td[:, ant_pos])) - 1e-9)
        if signal_peak > first_peak:
            shift = signal_peak - first_peak
            time_aligned_signals[:np.size(time_aligned_signals, axis=0)
                                  - shift[0], ant_pos] = \
                np.abs(td[shift[0]:, ant_pos])
        else:
            shift = first_peak - signal_peak
            time_aligned_signals[shift[0]:, ant_pos] = \
                np.abs(td[:np.size(time_aligned_signals, axis=0) - shift[0],
                       ant_pos])

    kernel = np.average(time_aligned_signals, axis=1)
    # kernel = avg_signal
    # average peak index for correlation data interpretation
    max_avg = np.max(kernel)
    avg_peak, _ = find_peaks(kernel, height=max_avg - 1e-9)
    previous_peak_idx = 0

    for ant_pos in range(np.size(td, axis=1)):
        # corresponding intensities
        position = np.abs(td[:, ant_pos])

        # correlation
        corr = signal.correlate(position, kernel, 'same')

        # resizing correlation back corresponding to the
        # antenna position array length
        lags = signal.correlation_lags(len(position), len(kernel), 'same')
        max_corr = np.max(corr)
        corr_peaks, _ = find_peaks(corr, height=max_corr - 1e-9)

        # positive index - average signal is "shifted" to the right wrt
        # the actual signal

        # negative - to the left
        indx_lag = lags[corr_peaks]

        # approximate anticipated peak index
        approx_peak_idx = avg_peak + indx_lag

        # store anticipated time responce
        # tr_approx = np.append(tr_approx, ts[approx_peak_idx])

        # find the closest actual peak index to approximate
        peaks, _ = find_peaks(position)
        peak = peaks[np.argmin(np.abs(peaks - approx_peak_idx))]

        if previous_peak_idx == 0 or np.abs(peak - previous_peak_idx) > \
                peak_threshold:
            peak = approx_peak_idx

        # store the time responce obtained from threshold method
        tr_threshold = np.append(tr_threshold, ts[peak])

        # polar radius of a corresponding highest intensity response
        # (corrected radius - radius of a time response)
        rad = ant_rad - ts[peak] * 1e-9 * __VAC_SPEED / 2
        # TODO: account for new antenna time delay

        previous_peak_idx = peak
        # appending polar radius to rho array
        rho = np.append(rho, rad)

        #
        # fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9, 10), dpi=100)
        #
        # ax1.plot(np.abs(position), 'b-')
        # ax1.plot(peaks, np.abs(position[peaks]), 'rx', label='Peaks')
        # ax1.plot(peak, np.abs(position[peak]), 'yx', label='Chosen peak')
        # ax1.plot(approx_peak_idx, np.abs(position[approx_peak_idx]), 'mx',
        #                                         label='Approximation response')
        # ax1.set_title('Antenna position slice')
        # ax2.plot(np.abs(kernel), 'r-')
        # ax2.plot(avg_peak, np.abs(kernel[avg_peak]), 'kx',
        #                         label='Average signal peak = %d' % avg_peak)
        # ax2.set_title('Average signal for all positions')
        # ax3.plot(lags, np.abs(corr), 'g-')
        # ax3.plot(indx_lag, np.abs(corr[corr_peaks]), 'cx',
        #     label='Correlation peak = %d\nThreshold: %d\nApproximation: %d'
        #           % (indx_lag, peak, approx_peak_idx))
        # # ax3.plot(np.abs(corr))
        # ax3.set_title('Correlation of both signals')
        # #
        # fig.legend()
        # plt.savefig(os.path.join(out_dir, 'slice_%d.png'
        #                                     %  ant_pos))
        # plt.close(fig)
        # plt.show()

    rho = np.flip(rho)
    # CubicSpline interpolation on a given set of data
    cs = polar_fit_cs(rho, angles)

    # calculate the center of mass fo this shape
    x_cm, y_cm = find_centre_of_mass(rho, angles)

    td_plt = td[:700, :]
    plt_extent = [0, 355, ts_plt[-1], ts_plt[0]]
    plt_aspect_ratio = 355 / ts_plt[-1]

    # Plot primary scatter forward projection only
    plt.figure()
    plt.rc('font', family='Times New Roman')
    plt.imshow(np.abs(td_plt), aspect=plt_aspect_ratio, cmap='inferno',
               extent=plt_extent)
    plt.colorbar(format='%.2e').ax.tick_params(labelsize=16)
    plt.gca().set_yticks([round(ii, 2)
                          for ii in ts[::200 // 8]])
    plt.gca().set_xticks([round(ii)
                          for ii in np.linspace(0, 355, 355)[::75]])
    plt.title('Boundary check', fontsize=20)
    plt.xlabel('Polar Angle of Antenna Position ('
               + r'$^\circ$' + ')',
               fontsize=16)
    plt.ylabel('Time of Response (ns)', fontsize=16)
    plt.plot(plt_angles, tr_threshold, 'r-', linewidth=1, label='Threshold')
    # plt.plot(plt_angles, tr_approx, 'r--', linewidth=1, label='Approximation')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boundary_vs_sino_noncrop.png'),
                dpi=300)

    # x = rho * np.cos(angles)
    # y = rho * np.sin(angles)
    #
    # x_diff = x - x_cm
    # y_diff = y - y_cm
    #
    # rho_rad = np.sqrt(x_diff ** 2 + y_diff ** 2)
    # avg = np.average(rho_rad)
    #
    # plt.plot(angles, rho_rad, 'ro', label='Distance to COM')
    # plt.plot(angles, rho_rad, 'r--')
    # plt.plot(angles, rho, 'bx', label=r'Distance to $(0,0)$')
    # plt.plot((angles[0], angles[-1]), (avg, avg), 'k-',
    #                                                   label='Average radius')
    # plt.plot((angles[0], angles[-1]), (phantom_rad, phantom_rad), 'b--',
    #                                                      label='Real radius')
    # plt.xlabel('Angular antenna position, rad')
    # plt.ylabel('Radius, m')
    # plt.legend()
    # plt.show()
    # delta_r = avg - phantom_rad
    # delta_r_dev = rho_rad - avg
    #
    # delta_r_neg = np.min(delta_r_dev)
    # delta_r_pos = np.max(delta_r_dev)
    #
    # deviation = np.average(delta_r_dev)
    # uns_dev = np.average(np.abs(delta_r_dev))

    return cs, x_cm, y_cm
