"""
Tyson Reimer
University of Manitoba
November 8, 2018
"""

import numpy as np
import scipy.constants
from scipy.optimize import leastsq, minimize

from umbms.beamform.breastmodels import get_breast, get_roi
from umbms.beamform.utility import (
    apply_ant_t_delay,
)

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
    """Calculates average breast propagation velocity
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
    avg_perm = ((100 - fibr_perc) / 100.0) * measured_adi_perm + (
        fibr_perc / 100.0
    ) * measured_fib_perm

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
    """Calculates propagation speed wrt every frequency and
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

    beta = (
        2
        * np.pi
        * freqs
        * np.sqrt(
            (__VAC_PERMEABILITY * __VAC_PERMITTIVITY * permittivities / 2)
            * (
                np.sqrt(
                    1
                    + (
                        conductivities
                        / (
                            2
                            * np.pi
                            * freqs
                            * permittivities
                            * __VAC_PERMITTIVITY
                        )
                    )
                    ** 2
                )
                + 1
            )
        )
    )

    return 2 * np.pi * freqs / beta


def get_speed_from_epsilon(epsilon):
    """Returns the propagations speed based on real and imaginary parts
    of the permittivity

    Parameters:
    --------------------
    epsilon : array_like
        Complex permittivity

    Returns:
    --------------------
    v : array_like
        Propagation speed
    """

    a = np.sqrt(1 + (np.imag(epsilon) / np.real(epsilon)) ** 2) + 1
    v = 1 / np.sqrt(
        __VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 * np.real(epsilon) * a
    )

    return v


def cole_cole(freq, e_h, e_s, tau, alpha):
    """Returns the complex permittivity modelled according to the
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
    return e_h + (e_s - e_h) / (
        1 + (2j * np.pi * freq * tau * 1e-12) ** (1 - alpha)
    )


def phase_shape(freq, length, epsilon, shift):
    """Returns the unwrapped shape of the phase based on the complex
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

    a = np.sqrt(1 + (np.imag(epsilon) / np.real(epsilon)) ** 2) + 1
    b = np.sqrt(
        __VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 * np.real(epsilon) * a
    )
    return -2 * np.pi * freq * length * b - shift


def phase_shape_average(freq, length, epsilon, shift, d_air, d):
    """Returns the unwrapped shape of the phase based on the complex
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

    a = np.sqrt(1 + (np.imag(epsilon) / np.real(epsilon)) ** 2) + 1
    b = np.sqrt(
        __VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 * np.real(epsilon) * a
    )
    v_avg = length / (2 * d_air / __VAC_SPEED + d * b)
    return -2 * np.pi * freq * length / v_avg - shift


def phase_shape_explicit(freq, length, e_h, e_s, tau, alpha, shift):
    """Returns the unwrapped shape of the phase based on the complex
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

    epsilon = cole_cole(freq, e_h, e_s, tau, alpha)
    a = np.sqrt(1 + (np.imag(epsilon) / np.real(epsilon)) ** 2) + 1
    b = np.sqrt(
        __VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 * np.real(epsilon) * a
    )
    return -2 * np.pi * freq * length * b - shift


def phase_shape_wrapped(freq, length, epsilon, shift):
    """Returns the wrapped shape of the phase based on the complex
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

    a = np.sqrt(1 + (np.imag(epsilon) / np.real(epsilon)) ** 2) + 1
    b = np.sqrt(
        __VAC_PERMEABILITY * __VAC_PERMITTIVITY / 2 * np.real(epsilon) * a
    )

    shape = -2 * np.pi * freq * length * b - shift
    # Wrap
    shape = (np.pi + shape) % (2 * np.pi) - np.pi
    return shape


def phase_diff_MSE(x, exp_phase, freq, length, wrapped=False, d_air=0, d=0):
    """Returns the MSE between the experimental phase
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
    elif d_air != 0 and d != 0:
        shape = phase_shape_average(freq, length, epsilon, shift, d_air, d)
    else:
        shape = phase_shape(freq, length, epsilon, shift)

    if exp_phase.ndim == 1:
        mse = (1 / np.size(exp_phase)) * (np.sum((exp_phase - shape) ** 2))
    else:
        mse = (1 / np.size(exp_phase)) * (
            np.sum(
                (exp_phase - shape.reshape(np.size(exp_phase, axis=0), 1)) ** 2
            )
        )

    return mse


def speed_diff_MSE(x, exp_speed, freq, length):
    """Returns the MSE between the experimental speed
    and the derived shape

    Parameters:
    ----------------
    x : array-like
        An array of parameters for the Cole-Cole model
    exp_speed :  array-like
        Extracted speed (m/s)
    freq : array-like
        Frequencies used in the experiment (Hz)
    length : float
        The transmission distance (separation between the antennas)

    Returns:
    ----------------
    mse : float
        Mean square error
    """

    (e_h, e_s, tau, alpha) = x
    epsilon = cole_cole(freq, e_h, e_s, tau, alpha)

    shape = get_speed_from_epsilon(epsilon)

    mse = (1 / np.size(exp_speed)) * (np.sum((exp_speed - shape) ** 2))

    return mse


def fit_bootstrap(p0, datax, datay, length, function, yerr_systematic=0.0):
    errfunc = lambda p, x, y, l: function(x, l, *p) - y

    # Fit first time
    pfit = leastsq(errfunc, p0, args=(datax, datay, length), full_output=False)[
        0
    ]

    # pfit = res.x
    # perr = res.cov_x

    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay, length)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(100):
        randomDelta = np.random.normal(0.0, sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit = leastsq(
            errfunc, p0, args=(datax, randomdataY, length), full_output=False
        )[0]

        # randomfit = randres.x
        ps.append(randomfit)

    ps = np.array(ps)
    mean_pfit = np.mean(ps, 0)

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    Nsigma = 2.0  # 1sigma gets approximately the same as methods above
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps, 0)

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap


def fit_bootstrap_minimize(
    p0, datax, datay, length, function, resfunc, yerr_systematic=0.0
):
    errfunc = lambda p, x, y, l: resfunc(x, l, *p) - y

    results = minimize(
        fun=function,
        x0=p0,
        bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25), (None, None)),
        args=(datay, datax, length, False),
        method="trust-constr",
        options={"maxiter": 2000},
    )

    pfit = results.x

    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay, length)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(10):
        randomDelta = np.random.normal(0.0, sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit = minimize(
            fun=function,
            x0=p0,
            bounds=((1, 7), (8, 80), (7, 103), (0.0, 0.25), (None, None)),
            args=(randomdataY, datax, length, False),
            method="trust-constr",
            options={"maxiter": 2000},
        )

        ps.append(randomfit.x)

    ps = np.array(ps)
    mean_pfit = np.mean(ps, 0)

    # You can choose the confidence interval that you want for your
    # parameter estimates:
    Nsigma = 2.0  # 1sigma gets approximately the same as methods above
    # 1sigma corresponds to 68.3% confidence interval
    # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps, 0)

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap
