"""
Prykhodko Illia
University of Manitoba
November 17, 2022
"""
import os
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy.interpolate import CubicSpline
from scipy.constants import speed_of_light
from umbms.beamform.iczt import iczt
from umbms.beamform.utility import get_xy_arrs, rect
from umbms.boundary.differential_minimization import shift_and_rotate
from umbms.plot.sinogramplot import show_sinogram
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


def shift_cs(cs, delta_x, delta_y):
    """Shifts the CubicSpline according to Cartesian shift in the image
    domain

    Parameters
    -----------
    cs : PPoly
        CubicSplice of a given boundary
    delta_x, delta_y : float
        Spatial shift

    Returns
    -----------
    cs_shifted : PPoly
        CubicSpline that corresponds to the spatial shift
    """
    angs = np.linspace(0, np.deg2rad(359), 300)
    rhos = cs(angs)

    xs = rhos * np.cos(angs)
    ys = rhos * np.sin(angs)

    xs_shifted = xs + delta_x
    ys_shifted = ys + delta_y

    rho_shifted, phi_shifted = cart_to_polar(xs_shifted, ys_shifted)
    cs_shifted = polar_fit_cs(rho_shifted, phi_shifted)

    return cs_shifted


def shift_rot_cs(cs, delta_x, delta_y, delta_phi):
    """Shifts and rotates the CubicSpline according to Cartesian shift
    in the image domain

    Parameters
    -----------
    cs : PPoly
        CubicSplice of a given boundary
    delta_x, delta_y : float
        Spatial shift
    delta_phi : float
        Spatial rotation

    Returns
    -----------
    cs_shifted : PPoly
        CubicSpline that corresponds to the spatial shift
    """

    angs = np.linspace(0, np.deg2rad(359), 300)
    rhos = cs(angs)

    xs = rhos * np.cos(angs)
    ys = rhos * np.sin(angs)

    xs_shift_rot, ys_shift_rot = shift_and_rotate(xs=xs, ys=ys,
                                                  delta_x=delta_x,
                                                  delta_y=delta_y,
                                                  delta_phi=delta_phi)

    rho_shift_rot, phi_shift_rot = cart_to_polar(xs_shift_rot, ys_shift_rot)
    cs_shifted = polar_fit_cs(rho_shift_rot, phi_shift_rot)

    return cs_shifted


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


def prepare_fd_data(adi_emp_cropped, ini_t, fin_t, n_time_pts, ini_f, fin_f,
                    temp=False):
    """Prepares unorganized data for boundary detection"""

    # convert frequency-domain data to time-domain
    td = iczt(adi_emp_cropped, ini_t=ini_t, fin_t=fin_t, n_time_pts=n_time_pts,
              ini_f=ini_f, fin_f=fin_f)
    # time response data
    ts = np.linspace(ini_t, fin_t, n_time_pts)
    # find the kernel
    kernel = time_aligned_kernel(td)
    return td, ts, kernel


def get_boundary_iczt(adi_emp_cropped, ant_rad, *, n_ant_pos=72,
                      ini_ant_ang=-136.0, ini_t=0.5e-9, fin_t=5.5e-9,
                      n_time_pts=700, ini_f=2e9, fin_f=9e9, peak_threshold=10,
                      plt_slices=False, plot_sino=False, out_dir='',
                      cs_shift=None):
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

    # Prepare the data to obtain time-domain scan data, time points and time
    # aligned kernel for cross-correlation
    td, ts, kernel = prepare_fd_data(adi_emp_cropped=adi_emp_cropped,
                                     ini_t=ini_t, fin_t=fin_t,
                                     n_time_pts=n_time_pts, ini_f=ini_f,
                                     fin_f=fin_f)

    # find all rho and ToR (time-of-response) values
    # on each antenna position
    rho, ToR = rho_ToR_from_td(td=td, ts=ts, kernel=kernel, ant_rad=ant_rad,
                               peak_threshold=peak_threshold,
                               plt_slices=plt_slices, out_dir=out_dir)

    # polar angle data
    angles = np.linspace(0, np.deg2rad(355), n_ant_pos) \
             + np.deg2rad(ini_ant_ang)
    # CubicSpline interpolation on a given set of data
    cs = polar_fit_cs(rho, angles)

    # calculate the center of mass fo this shape
    x_cm, y_cm = find_centre_of_mass(rho, angles)

    if plot_sino:
        # angles for plotting
        plt_angles = np.linspace(0, 355, n_ant_pos)
        ts_plt = ts[:n_time_pts]

        td_plt = td[:, :]
        plt_extent = [0, 355, ts_plt[-1], ts_plt[0]]
        plt_aspect_ratio = 355 / ts_plt[-1]

        show_sinogram(data=td_plt, aspect_ratio=plt_aspect_ratio,
                      extent=plt_extent, title='Boundary check',
                      out_dir=out_dir,
                      save_str='boundary_vs_sino_no_shift.png',
                      ts=ts_plt, transparent=False, bound_angles=plt_angles,
                      bound_times=ToR, bound_color='r')

        if cs_shift is not None:
            xs_shifted = rho * np.cos(angles) + cs_shift[0]
            ys_shifted = rho * np.sin(angles) + cs_shift[1]
            rho_shifted, phi_shifted = cart_to_polar(xs_shifted, ys_shifted)
            tr_shifted = (2 / __VAC_SPEED) * (ant_rad - rho_shifted) * 1e9

            show_sinogram(data=td_plt, aspect_ratio=plt_aspect_ratio,
                          extent=plt_extent, title='Boundary check',
                          out_dir=out_dir,
                          save_str='boundary_vs_sino_shift.png',
                          ts=ts_plt, transparent=False,
                          bound_angles=plt_angles,
                          bound_times=tr_shifted, bound_color='b')

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


def time_aligned_kernel(td):
    """Produces a kernel for cross-correlation by time-aligning most
    intense peaks

    Parameters
    ----------
    td : array_like
        Skin response array in time domain

    Returns
    ----------
    kernel : array_like
        Kernel for cross-correlation
    """

    # Initialize the array to store aligned responses
    time_aligned_signals = np.zeros_like(td, dtype=float)
    # Aligning wrt to the first antenna position
    time_aligned_signals[:, 0] = np.abs(td[:, 0])
    # Find the time of response for the first antenna position
    first_peak, _ = find_peaks(time_aligned_signals[:, 0],
                               height=np.max(
                                   time_aligned_signals[:, 0]) - 1e-9)

    # for each antenna position
    for ant_pos in range(1, np.size(td, axis=1), 1):
        # find the max intensity ToR
        signal_peak, _ = find_peaks(np.abs(td[:, ant_pos]),
                                    np.max(np.abs(td[:, ant_pos])) - 1e-9)
        # if the ToR is later
        if signal_peak > first_peak:
            shift = signal_peak - first_peak
            # then on this position fill from the start of the array to
            # the position *shift* units back from the end
            time_aligned_signals[:np.size(time_aligned_signals, axis=0)
                                  - shift[0], ant_pos] = \
                np.abs(td[shift[0]:, ant_pos])
        # if the ToR is earlier
        else:
            shift = first_peak - signal_peak
            # then on this position fill from the *shift* index to the
            # end of the array
            time_aligned_signals[shift[0]:, ant_pos] = \
                np.abs(td[:np.size(time_aligned_signals, axis=0) - shift[0],
                       ant_pos])

    # obtain the kernel by averaging
    kernel = np.average(time_aligned_signals, axis=1)
    return kernel


def rho_ToR_from_td(td, ts, kernel, ant_rad, peak_threshold,
                    plt_slices, out_dir):
    """Completes the CMPPS routine, returns the breast phantom boundary
    as an array of rho and time responses

    Parameters
    ----------
    td : array_like
        Scan data in time domain
    ts : array_like
        Time points used in the sinogram
    kernel : array_like
        Kernel for cross-correlation
    ant_rad : float
         Radius of antenna position AFTER all the corrections
    peak_threshold : int
        The maximal allowed index jump in time points array
        (recommended not to change)
    plt_slices : bool
        Flag to plot single antenna slices
    out_dir : str
         Directory to save slices

    Returns
    ----------
    rho : array_like
        Array of rho values at each antenna position (m)
    ToR : array_like
        Array of values of the time-of-response for every antenna
        position (s)
    """
    # creating an array of polar distances for storing
    rho = np.array([])
    # initializing an array of time-responces
    ToR = np.array([])

    # average peak index for correlation data interpretation
    max_avg = np.max(kernel)
    avg_peak, _ = find_peaks(kernel, height=max_avg - 1e-9)
    previous_peak_idx = 0

    for ant_pos in range(np.size(td, axis=1)):

        # -------PEAK SEARCH--------- #

        # corresponding intensities
        position = np.abs(td[:, ant_pos])
        # correlation
        corr = correlate(position, kernel, 'same')
        # resizing correlation back corresponding to the
        # antenna position array length
        lags = correlation_lags(len(position), len(kernel), 'same')
        max_corr = np.max(corr)
        corr_peaks, _ = find_peaks(corr, height=max_corr - 1e-9)

        # ------PEAK SELECTION------- #

        # positive index - average signal is "shifted" to the right wrt
        # the actual signal
        # negative - to the left
        indx_lag = lags[corr_peaks]
        # approximate anticipated peak index
        approx_peak_idx = avg_peak + indx_lag
        # find the closest actual peak index to approximate
        peaks, _ = find_peaks(position)
        peak = peaks[np.argmin(np.abs(peaks - approx_peak_idx))]
        # Correlation-matching peak proximity selection (CMPPS)
        if previous_peak_idx == 0 or np.abs(peak - previous_peak_idx) > \
                peak_threshold:
            peak = approx_peak_idx

        # --------POLAR RADIUS------- #

        # store the time response obtained from CMPPS method
        ToR = np.append(ToR, ts[peak])
        # polar radius of a corresponding highest intensity response
        # (corrected radius - radius of a time response)
        rad = ant_rad - ts[peak] * __VAC_SPEED / 2
        # TODO: account for new antenna time delay
        # appending polar radius to rho array
        rho = np.append(rho, rad)

        if plt_slices:
            plot_slices(position, peaks, peak, approx_peak_idx, kernel,
                        avg_peak, lags, corr, corr_peaks, indx_lag, ant_pos,
                        out_dir)

        # for next iteration
        previous_peak_idx = peak

    # the rotation is counter-clockwise
    rho = np.flip(rho)

    return rho, ToR


def plot_slices(position, peaks, peak, approx_peak_idx, kernel, avg_peak, lags,
                corr, corr_peaks, indx_lag, ant_pos, out_dir):
    """Produces the plot of the antenna position slice with a kernel and
    a result of correlation

    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9, 10), dpi=100)

    ax1.plot(np.abs(position), 'b-')
    ax1.plot(peaks, np.abs(position[peaks]), 'rx', label='Peaks')
    ax1.plot(peak, np.abs(position[peak]), 'yx', label='Chosen peak')
    ax1.plot(approx_peak_idx, np.abs(position[approx_peak_idx]), 'mx',
             label='Approximation '
                   'response')
    ax1.set_title('Antenna position slice')
    ax2.plot(np.abs(kernel), 'r-')
    ax2.plot(avg_peak, np.abs(kernel[avg_peak]), 'kx',
             label='Average signal peak = %d' %
                   avg_peak)
    ax2.set_title('Average signal for all positions')
    ax3.plot(lags, np.abs(corr), 'g-')
    ax3.plot(indx_lag, np.abs(corr[corr_peaks]), 'cx',
             label='Correlation peak = %d\nCMPPS: %d\nApproximation: %d'
                   % (indx_lag, peak, approx_peak_idx))
    ax3.set_title('Correlation of both signals')
    fig.legend()
    plt.savefig(os.path.join(out_dir, 'slice_%d.png' % ant_pos))
    plt.close(fig)


def retain_and_tile_freqs(scan_ini_f, scan_fin_f, ini_f, n_fs, n_ant_pos):
    """Creates a tiled array of frequencies for phase shifting.

    The frequencies are created in the range [scan_ini_f, scan_fin_f]
    with a step size n_fs.

    The frequencies are retained in the range [ini_f, scan_fin_f] on a
    condition ini_f >= scan_ini_f

    The frequencies are tiled n_ant_pos times.

    Parameters
    -----------
    scan_ini_f : float
        Starting frequency of the scan
    scan_fin_f : float
        Finishing frequency of the scan
    ini_f : float
        Frequency to retain (higher than)
    n_fs : int
        Step size
    n_ant_pos : int
        Number of antenna positions during the scan

    Returns
    ------------
    freqs : array_like
        The tiled array
    """

    # Obtain frequency array for phase shifting
    fs = np.linspace(scan_ini_f, scan_fin_f, n_fs)
    fs = fs[fs >= ini_f]
    freqs = np.tile(fs, (n_ant_pos, 1)).T

    return freqs


def phase_shift(fd, delta_t, freqs):
    """Preforms a phase shift on the frequency domain data

    Parameters
    ------------
    fd : array_like
        Frequency domain data of the scan
    delta_t : array_like
        Time shifts for every antenna position
    freqs : array_like
        Frequencies used in the scan (tiled for more efficient
        computation)

    Returns
    ------------
    shifted_fd : array_like
        Frequency domain data after phase shifting
    """

    # exponential factor for Fourier transform
    exp_conversion = np.exp(-2j * np.pi * delta_t * freqs)
    # conversion
    shifted_fd = fd * exp_conversion

    return shifted_fd


def fd_differential_align(fd_emp_ref_left, fd_emp_ref_right, ant_rad=0.0,
                          ini_t=0.5e-9, fin_t=5.5e-9, n_time_pts=700,
                          ini_f=2e9, fin_f=9e9, peak_threshold=10,
                          scan_ini_f=1e9, scan_fin_f=9e9, n_fs=1001):
    """Converts aligned time-responses into frequency domain

    https://en.wikipedia.org/wiki/Fourier_transform
    Performs a phase shift wrt the time shift per antenna position

    Assuming the left breast is fixed

    Parameters
    ----------
    fd_emp_ref_left : array_like
        FD data of the left breast (empty chamber reference)
    fd_emp_ref_right : array_like
        FD data of the right breast (empty chamber reference)
    ant_rad : float
        Radius of antenna position AFTER all the corrections
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        (s)
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        (s)
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The smallest frequency to be retained, (Hz)
    fin_f : float
        The largest frequency to be retained, (Hz)
    peak_threshold : int
        The maximal allowed index jump in time points array
        (recommended not to change)
    scan_ini_f : int
        The first frequency of the scan (Hz)
    scan_fin_f : int
        The last frequency of the scan (Hz)
    n_fs : int
        Number of frequency steps
    Returns
    ----------
    s11_aligned_right :  array_like
        FD data that corresponds to spatial alignment of the right
        breast wrt to the left
    """

    # Number of antenna positions
    n_ant_pos = np.size(fd_emp_ref_left, axis=1)
    # Get frequencies for phase shifting
    freqs = retain_and_tile_freqs(scan_ini_f=scan_ini_f,
                                  scan_fin_f=scan_fin_f, ini_f=ini_f,
                                  n_fs=n_fs, n_ant_pos=n_ant_pos)
    # convert FD to TD
    td_left = iczt(fd_emp_ref_left, ini_t=ini_t, fin_t=fin_t,
                   n_time_pts=n_time_pts, ini_f=ini_f, fin_f=fin_f)
    td_right = iczt(fd_emp_ref_right, ini_t=ini_t, fin_t=fin_t,
                    n_time_pts=n_time_pts, ini_f=ini_f, fin_f=fin_f)

    # time response data
    ts = np.linspace(ini_t, fin_t, n_time_pts)

    # find the kernels
    kernel_left = time_aligned_kernel(td_left)
    kernel_right = time_aligned_kernel(td_right)

    # find all ToR values on each antenna position
    _, ToR_left = rho_ToR_from_td(td=td_left, ts=ts, kernel=kernel_left,
                                  ant_rad=ant_rad,
                                  peak_threshold=peak_threshold,
                                  plt_slices=False, out_dir='')
    _, ToR_right = rho_ToR_from_td(td=td_right, ts=ts, kernel=kernel_right,
                                   ant_rad=ant_rad,
                                   peak_threshold=peak_threshold,
                                   plt_slices=False, out_dir='')
    # time shifts
    delta_t = (ToR_left - ToR_right)

    # delta1 = delta_t[:36]
    # delta2 = delta_t[36:]
    # diff = np.abs(delta1 - delta2)

    # Phase shift
    s11_aligned_right = phase_shift(fd=fd_emp_ref_right, delta_t=delta_t,
                                    freqs=freqs)

    return s11_aligned_right


def extract_delta_t_from_boundary(tor_right, cs_right_shifted, ant_rad,
                                  n_ant_pos=72, ini_ant_ang=-136.):
    """

    Parameters
    ----------
    tor_right : array_like
        An array of times-of-response for an unshifted right boundary
    cs_right_shifted : PPoly
        Shifted CubicSpline of the right breast to align with the left
    ant_rad : float
        Corrected antenna radius

    Returns
    ----------
    delta_t : array_like
        An array of time differences for further phase shift
    """

    # angles - real angles of antenna
    ant_angs = np.linspace(0, np.deg2rad(355), n_ant_pos) + \
               np.deg2rad(ini_ant_ang)
    ant_angs = np.flip(ant_angs)

    cs_angs = np.linspace(0, np.deg2rad(360), 1000)
    rhos = cs_right_shifted(cs_angs)

    # antenna positions
    ant_xs = ant_rad * np.cos(ant_angs)
    ant_ys = ant_rad * np.sin(ant_angs)

    # coordinates of the breast outline
    breast_xs = rhos * np.cos(cs_angs)
    breast_ys = rhos * np.sin(cs_angs)

    # distances from antennas to the phantom
    distances = np.sqrt((ant_xs[:, None] - breast_xs) ** 2 +
                        (ant_ys[:, None] - breast_ys) ** 2)

    min_dists = np.min(distances, axis=-1)
    # convert distances to time of flight
    tor_shifted = min_dists / __VAC_SPEED * 2
    delta_t = tor_shifted - tor_right

    return delta_t


def phase_shift_aligned_boundaries(fd_emp_ref_right, ant_rad, cs_right_shifted,
                                   ini_t, fin_t, n_time_pts, ini_f, fin_f,
                                   n_fs, scan_ini_f, scan_fin_f):
    """Performs phase shifting procedure based on the spatial shift
    of the boundary

    Parameters
    -----------
    fd_emp_ref_right : array_like
        Frequency domain data of the unshifted right breast scan
        (empty chamber reference)
    ant_rad : float
        Radius of antenna AFTER applying the corrections
    cs_right_shifted : PPoly
        CubicSpline of the shifted right breast boundary
    ini_t : float
        Initial time point for ICZT (s)
    fin_t : float
        Finishing time point for ICZT (s)
    n_time_pts : int
        Number of time points
    ini_f : float
        Initial frequency that retains (Hz)
    fin_f : float
        Finishing frequency (Hz)
    n_fs : int
        Number of frequencies
    scan_ini_f : float
        Initial frequency of the scan (Hz)
    scan_fin_f : float
        Finishing frequency of the scan (Hz)

    Returns
    ---------
    s11_aligned_right : array_like
        Resulting S11 data after the application of the phase shift
    """

    # if regular scan (up to 9GHz)
    if scan_fin_f is None:
        # the range to retain and the range of the scan end at the same
        # frequency
        scan_fin_f = fin_f

    # Number of antenna positions
    n_ant_pos = np.size(fd_emp_ref_right, axis=1)

    #
    td, ts, kernel = prepare_fd_data(
        adi_emp_cropped=fd_emp_ref_right,
        ini_t=ini_t, fin_t=fin_t, n_time_pts=n_time_pts,
        ini_f=ini_f, fin_f=fin_f)

    # Prepare the data to obtain time-domain scan data, time points and
    # time aligned kernel for cross-correlation
    _, tor_right = rho_ToR_from_td(td, ts, kernel,
                                   ant_rad=ant_rad, peak_threshold=10,
                                   plt_slices=False, out_dir='')

    # Get frequencies for phase shifting
    freqs = retain_and_tile_freqs(scan_ini_f=scan_ini_f,
                                  scan_fin_f=scan_fin_f,
                                  n_fs=n_fs, ini_f=ini_f,
                                  n_ant_pos=n_ant_pos)

    # Find delta_t values that correspond to a spatial shift
    # of the boundary
    delta_t = extract_delta_t_from_boundary(tor_right=tor_right,
                                            cs_right_shifted=cs_right_shifted,
                                            ant_rad=ant_rad)

    # Phase shift
    s11_aligned_right = phase_shift(fd=fd_emp_ref_right, delta_t=delta_t,
                                    freqs=freqs)

    return s11_aligned_right


def window_skin_alignment(fd_emp_ref_right, fd_emp_ref_left, ant_rad=0.0,
                          ini_t=0.5e-9, fin_t=5.5e-9, n_time_pts=700,
                          ini_f=2e9, fin_f=9e9, peak_threshold=10,
                          scan_ini_f=1e9, scan_fin_f=9e9, n_fs=1001):
    """Shifts the skin response in a thin window defined by the earliest
    and latest skin signals


    Parameters
    ----------
    fd_emp_ref_left : array_like
        FD data of the left breast (empty chamber reference)
    fd_emp_ref_right : array_like
        FD data of the right breast (empty chamber reference)
    ant_rad : float
        Radius of antenna position AFTER all the corrections
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        (s)
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        (s)
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The smallest frequency to be retained, (Hz)
    fin_f : float
        The largest frequency to be retained, (Hz)
    peak_threshold : int
        The maximal allowed index jump in time points array
        (recommended not to change)
    scan_ini_f : int
        The first frequency of the scan (Hz)
    scan_fin_f : int
        The last frequency of the scan (Hz)
    n_fs : int
        Number of frequency steps
    """

    # convert FD to TD
    td_left = iczt(fd_emp_ref_left, ini_t=ini_t, fin_t=fin_t,
                   n_time_pts=n_time_pts, ini_f=ini_f, fin_f=fin_f)
    td_right = iczt(fd_emp_ref_right, ini_t=ini_t, fin_t=fin_t,
                    n_time_pts=n_time_pts, ini_f=ini_f, fin_f=fin_f)

    # time response data
    ts = np.linspace(ini_t, fin_t, n_time_pts)

    # find the kernels
    kernel_left = time_aligned_kernel(td_left)
    kernel_right = time_aligned_kernel(td_right)

    # find all ToR values on each antenna position
    _, ToR_left = rho_ToR_from_td(td=td_left, ts=ts, kernel=kernel_left,
                                  ant_rad=ant_rad,
                                  peak_threshold=peak_threshold,
                                  plt_slices=False, out_dir='')
    _, ToR_right = rho_ToR_from_td(td=td_right, ts=ts, kernel=kernel_right,
                                   ant_rad=ant_rad,
                                   peak_threshold=peak_threshold,
                                   plt_slices=False, out_dir='')

    # Find time shifts for the skin
    delta_t = ToR_left - ToR_right

    # Frequencies used in the scan
    freqs = np.linspace(scan_ini_f, scan_fin_f, n_fs)
    freqs = freqs[freqs >= ini_f]
    df = np.diff(freqs)[0]

    # find a frequency that is the closest positive to zero
    # with a step size = df
    f_0 = freqs[0] - df * (freqs[0]//df)

    # Arrange an array of the size *7000* (a very large array)
    freqs_to_convolve = np.arange(f_0 - 3500 * df, f_0 + 3500 * df, df)

    # Extract the time point to ensure no loss in data (TD -> FD)
    ifft_ts = np.fft.fftfreq(n=np.size(freqs_to_convolve), d=df)
    t_0 = np.min(ifft_ts)
    t_fin = np.max(ifft_ts)

    new_right_breast_fd = np.empty_like(fd_emp_ref_right, dtype=complex)

    for ii in range(np.size(fd_emp_ref_right, axis=1)):  # for each antenna pos

        # ================ EXTRACTING THE SKIN WINDOW ================ #
        signal_td = np.abs(td_right[:, ii])
        # Searching for the peaks of the vertically flipped signal
        # to capture the whole width of the skin peak
        peaks, _ = find_peaks(-signal_td)
        # Maximal peak refers to a primary skin response (it might not be
        # of maximal intensity
        max_peak = np.argwhere(ts == ToR_right[ii])[0]
        # The start and the end of the skin are two closest negative
        # peaks to the primary skin response on either side
        peaks_to_decide = peaks - max_peak
        t_start = ts[max_peak + peaks_to_decide[peaks_to_decide < 0][-1]]
        t_end = ts[max_peak + peaks_to_decide[peaks_to_decide > 0][0]]

        # ============= PREPARE DATA FOR THE CONVOLUTION ============= #
        # Zero-pad the signal, so it can be easily extracted from the
        # result of the convolution
        original_signal_zero_padded = np.zeros_like(freqs_to_convolve,
                                                    dtype=complex)
        # Start at the index that corresponds to frequency ini_f in
        # a large array of frequencies
        start_freq_for_padding = int(np.size(freqs_to_convolve) // 2 - 1 +\
                                 freqs[0] // df)
        # Assign the signal to this region
        original_signal_zero_padded[start_freq_for_padding:
                                    start_freq_for_padding+np.size(
                                        fd_emp_ref_right, axis=0)] = \
            fd_emp_ref_right[:, ii]

        # ============= APPLY THE FOURIER TRANSFORM RULES ============ #
        # S11 * sinc for the pre-skin region
        pre_skin_sinc_conv = np.convolve(
            original_signal_zero_padded, (t_start - t_0) * np.sinc(
                freqs_to_convolve * (t_start - t_0)) * np.exp(-2j * np.pi *
                    freqs_to_convolve * (t_0 + (t_start - t_0)/2)), mode='same')

        # S11 * sinc for the skin region * phase shift based on the
        # delta_t value for a given antenna position
        skin_sinc_conv = np.convolve(
            original_signal_zero_padded, (t_end - t_start) * np.sinc(
                freqs_to_convolve * (t_end - t_start)) * np.exp(-2j * np.pi *
                    freqs_to_convolve * (t_start + (t_end - t_start) / 2)),
            mode='same') * np.exp(-2j * np.pi * freqs_to_convolve * delta_t[
            ii])

        # S11 * sinc for after the skin region
        post_skin_sinc_conv = np.convolve(
            original_signal_zero_padded, (t_fin - t_end) * np.sinc(
                freqs_to_convolve * (t_fin - t_end)) * np.exp(-2j * np.pi *
                    freqs_to_convolve * (t_end + (t_fin - t_end) / 2)),
            mode='same')

        # Times df due to a large step size and a discrepancy between a
        # discrete and continuous convolution
        signal_composed = (pre_skin_sinc_conv + skin_sinc_conv + \
                                      post_skin_sinc_conv) * df

        new_right_breast_fd[:, ii] = signal_composed[
                                     start_freq_for_padding + 1:
                                     start_freq_for_padding + np.size(
                                        fd_emp_ref_right, axis=0) + 1]

        # Uncomment below to compare the left and right breast signals
        # at every antenna position

        # signal_to_plot = iczt(fd_data=new_right_breast_fd[:, ii],
        #                       ini_t=ini_t, fin_t=fin_t,
        #                       n_time_pts=n_time_pts,
        #                       ini_f=ini_f, fin_f=fin_f)
        # plt.plot(ts, td_left[:, ii], 'k-')
        # plt.plot(ts, signal_to_plot, 'r--')
        # plt.show()

    return new_right_breast_fd
