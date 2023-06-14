"""
Tyson Reimer
University of Manitoba
November 8 2018
"""

import numpy as np
import scipy.constants

from umbms.beamform.breastmodels import get_breast, get_roi
from umbms.beamform.utility import get_xy_arrs, get_pixdist_ratio, \
    apply_ant_t_delay, get_ant_scan_xys, get_ant_xy_idxs

###############################################################################

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__VAC_PERMITTIVITY = 8.85e-12
__VAC_PERMEABILITY = 1.25e-6

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
        Conductivities fitted to a Cole-Coel model

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
