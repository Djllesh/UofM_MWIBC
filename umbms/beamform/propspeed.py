"""
Tyson Reimer
University of Manitoba
November 8, 2018
"""

import numpy as np
import scipy.constants

from umbms.beamform.breastmodels import get_breast, get_roi
from umbms.beamform.utility import get_xy_arrs, get_pixdist_ratio, \
    apply_ant_t_delay, get_ant_scan_xys, get_ant_xy_idxs

###############################################################################

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__VAC_PERMITTIVITY = scipy.constants.epsilon_0
__VAC_PERMEABILITY = scipy.constants.mu_0

# Permittivity of the breast tissue analogs used in the lab at the
# central frequency (glycerin for fat, 30% Triton X-100 solution for
# fibroglandular, and saline solution for tumor)
measured_air_perm = 1
measured_adi_perm = 7.08
measured_fib_perm = 44.94
measured_tum_perm = 77.11


###############################################################################


def estimate_speed(adi_rad, ant_rad, m_size=150, new_ant=True):
    """Estimates the propagation speed of the signal in the scan

    Estimates the propagation speed of the microwave signal for
    *all* antenna positions in the scan. Estimates using the average
    propagation speed.

    Parameters
    ----------
    adi_rad : float
        The approximate radius of the breast, in m
    ant_rad : float
        The radius of the antenna trajectory in the scan, as measured
        from the black line on the antenna holder, in m
    m_size : int
        The number of pixels along one dimension used to model the 2D
        imaging chamber
    new_ant : bool
        If True, indicates the 'new' antenna (from 2021) was used
    Returns
    -------
    speed : float
        The estimated propagation speed of the signal at all antenna
        positions, in m/s
    """

    # Correct for antenna phase-delay
    ant_rad = apply_ant_t_delay(ant_rad, new_ant=new_ant)

    # Model the breast as a homogeneous adipose circle, in air
    breast_model = get_breast(m_size=m_size, adi_rad=adi_rad, ant_rad=ant_rad)

    # Get the region of the scan area within the antenna trajectory
    roi = get_roi(ant_rad, m_size, ant_rad)

    # Estimate the speed
    speed = np.mean(__VAC_SPEED / np.sqrt(breast_model[roi]))

    return speed


def get_breast_speed(fibr_perc):
    """ Calculates average breast propagation velocity
    for a given fibroglandular tissue percentage

    Parameters
    -----------
    fibr_perc : integer
        Percentage of fibroglandular tissue in a breast

    Returns
    -----------
    breast_speed : float
        Estimated average propagation velocity
    """

    # calculate average permittivity for a given percentage
    avg_perm = ((100 - fibr_perc) / 100.) * measured_adi_perm +\
               (fibr_perc / 100.) * measured_fib_perm

    breast_speed = __VAC_SPEED / np.sqrt(avg_perm)

    if breast_speed < __VAC_SPEED:
        return breast_speed


def get_speed_from_perm(permittivity):
    """
    Parameters
    -----------
    permittivity : float
        Relative permittivity

    Returns
    -----------
    breast_speed : float
        Propagation velocity in a space with given perm
    """

    breast_speed = __VAC_SPEED / np.sqrt(permittivity)
    return breast_speed


def get_breast_speed_freq(freqs, permittivities, conductivities):
    """ Calculates propagation speed wrt every frequency and
    corresponding permittivities and conductivities

    Parameters
    -----------
    freqs : array_like 1001 X 1
        Linearly spaced frequencies
    permittivities : array_like 1001 X 1
        Permittivities fitted to a Cole-Cole model
    conductivities : array_like 1001 X 1
        Conductivities fitted to a Cole-Cole model

    Returns
    -----------
    breast_speed : array_like 1001 X 1
        Estimated propagation velocities
    """

    beta = 2 * np.pi * freqs * np.sqrt((__VAC_PERMEABILITY * __VAC_PERMITTIVITY
                                        * permittivities / 2) * (np.sqrt(1
                                        + (conductivities / ( 2 * np.pi * freqs
                                        * permittivities * __VAC_PERMITTIVITY))
                                                                    ** 2) + 1))

    return 2 * np.pi * freqs / beta


def cole_cole(freq, e_h, e_s, tau, alpha):
    """ Returns the complex permittivity modelled according to the
    Cole-Cole model : https://en.wikipedia.org/wiki/Cole%E2%80%93Cole_equation

    Parameters:
    -----------------------
    freq : array_like
        Frequencies
    e_h : float
        'Infinite frequency' dielectric constant
    e_s : float
        'Static frequency' dielectric constant
    tau : float
        Dielectric relaxation time constant
    alpha : float
        Exponent parameter

    Returns:
    ----------------------
    epsilon : array_like
        Complex permittivity
    """
    return e_h + (e_s - e_h)/(1 + (2j * np.pi * freq * tau * 1e-12)**
                              (1 - alpha))


def phase_shape(freq, length, epsilon, shift):
    """ Returns the unwrapped shape of the phase based on the complex
    permittivity

    Parameters:
    ------------------
    freq : array_like
        Frequencies
    length : float (m)
        Separation between the antennas
    epsilon : array_like
        Complex permittivity
    shift : float (rad)
        Vertical shift of the phase

    Returns:
    phase : array_like
        The unwrapped phase
    """

    a = np.sqrt(1 + (np.imag(epsilon)/np.real(epsilon))**2) + 1
    b = np.sqrt(__VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 *
                np.real(epsilon) * a)
    return - 2 * np.pi * freq * length * b - shift


def phase_shape_wrapped(freq, length, epsilon, shift):
    """ Returns the wrapped shape of the phase based on the complex
    permittivity

    Parameters:
    ------------------
    freq : array_like
        Frequencies
    length : float (m)
        Separation between the antennas
    epsilon : array_like
        Complex permittivity
    shift : float (rad)
        Vertical shift of the phase

    Returns:
    --------------------
    phase : array_like
        The wrapped phase
    """

    a = np.sqrt(1 + (np.imag(epsilon)/np.real(epsilon))**2) + 1
    b = np.sqrt(__VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 *
                np.real(epsilon) * a)

    shape = - 2 * np.pi * freq * length * b - shift
    # Wrap
    shape = (np.pi + shape) % (2 * np.pi) - np.pi
    return shape


def phase_diff_MSE(x, exp_phase, freq, length, wrapped=False):
    """ Returns the MSE between the experimental phase
    and the derived shape

    Parameters:
    ----------------
    x : array-like
        An array of parameters for the Cole-Cole model
    exp_phase :  array-like
        Extracted phase (rad)
    freq : array-like
        Frequencies used in the experiment (Hz)
    length : float
        The transmission distance (separation between the antennas)

    Returns:
    ----------------
    mse : float
        Mean square error
    """

    (e_h, e_s, tau, alpha, shift) = x
    epsilon = cole_cole(freq, e_h, e_s, tau, alpha)
    # Introducing the shift to not change the shape
    if wrapped:
        shape = phase_shape_wrapped(freq, length, epsilon, shift)
    else:
        shape = phase_shape(freq, length, epsilon, shift)

    mse = (1/np.size(exp_phase)) * (np.sum((exp_phase - shape)**2))
    return mse
