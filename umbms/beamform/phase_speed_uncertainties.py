"""
Illia Prykhodko

University of Manitoba
December 13th 2024
"""

import numpy as np
import scipy.constants

# Define propagation speed in vacuum
__VAC_SPEED = scipy.constants.speed_of_light
__VAC_PERMITTIVITY = scipy.constants.epsilon_0
__VAC_PERMEABILITY = scipy.constants.mu_0

def d_eprime_e_s(freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to e_s

    -------

    """
    omega = 2 * np.pi * freqs
    return ((1 + (omega * tau)**(1 - alpha)*np.sin(alpha * np.pi /2)) /
            (1 + 2*(omega * tau)**(1-alpha)*np.sin(alpha * np.pi/2) +
              (omega * tau)**(2*(1 - alpha))))


def d_eprime_e_h(freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to e_h

    -------

    """
    omega = 2 * np.pi * freqs
    return  1 - ((1 + (omega * tau) ** (1 - alpha) * np.sin(alpha * np.pi / 2))
            / (1 + 2 * (omega * tau) ** (1 - alpha) * np.sin(alpha * np.pi /
                                                            2) +
               (omega * tau) ** (2 * (1 - alpha))))


def d_eprime_alpha(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to alpha

    -------

    """
    omega = 2 * np.pi * freqs
    return ((e_h - e_s)*(omega* tau)**(alpha + 2)*(np.pi * tau * omega *
            np.cos(alpha * np.pi / 2)*(omega**2 * tau**2 - (omega * tau)**
            (2*alpha)) + 2 * np.log(omega * tau)*(omega * tau *
            np.sin(alpha * np.pi / 2)*(3 * (omega * tau)**(2 * alpha) - (
            omega * tau)**2) + 2 * (omega * tau)**(3 * alpha)))) /\
            (2 * (2 * omega**3 * tau**3*np.sin(alpha * np.pi / 2) +
            (omega * tau)**(3 * alpha) + (omega * tau)**(alpha + 2))**2)


def d_eprime_tau(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to tau

    -------

    """
    omega = 2 * np.pi * freqs
    return ((alpha - 1)*omega*(e_h - e_s)*(omega * tau)**(alpha + 1)*
            (omega * tau * np.sin(alpha * np.pi / 2))*(3*(omega * tau)**
            (2 * alpha) - (omega * tau)**2) + 2 * (omega * tau)**(3 * alpha))/\
            (2 * (omega * tau)**3 * np.sin(alpha * np.pi / 2) +
            (omega * tau)**(3 * alpha) + (omega * tau)**(alpha + 2))**2


def d_e2prime_e_s(freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to e_s

    -------

    """
    omega = 2 * np.pi * freqs
    return (((omega * tau)**(1 - alpha)*np.cos(alpha * np.pi /2)) /
            (1 + 2*(omega * tau)**(1-alpha)*np.sin(alpha * np.pi/2) +
             (omega * tau)**(2*(1 - alpha))))


def d_e2prime_e_h(freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to e_h

    -------

    """
    omega = 2 * np.pi * freqs
    return  - (((omega * tau) ** (1 - alpha) * np.cos(alpha * np.pi / 2))
                 / (1 + 2 * (omega * tau) ** (1 - alpha) * np.sin(alpha * np.pi /
                  2) + (omega * tau) ** (2 * (1 - alpha))))


def d_e2prime_alpha(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to alpha

    -------

    """
    omega = 2 * np.pi * freqs
    return ((omega * tau) ** 3 * (e_h - e_s) * (np.pi * np.sin(alpha * np.pi
        /2) * (2 * (omega * tau)**3*np.sin(alpha * np.pi / 2) + (
        omega * tau)**(3 * alpha) + (omega * tau)**(alpha + 2)) +
        2 * np.pi * (omega * tau) ** 3 * (np.cos(alpha * np.pi / 2)) ** 2 +
        2 * np.cos(alpha * np.pi / 2) * (3 * (omega * tau)**(2 * alpha)
        + (omega * tau)**2) * (omega * tau) ** alpha * np.log(omega * tau)))/\
        (2*(2*(omega * tau)**3*np.sin(alpha * np.pi / 2) + (omega * tau)**(3
         * alpha) + (omega * tau)**(alpha+2))**2)

def d_e2prime_tau(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to tau

    -------

    """
    omega = 2 * np.pi * freqs
    return ((alpha - 1)*omega**3*tau**2*np.cos(alpha * np.pi / 2)*(e_h - e_s)
            *(omega * tau)**alpha*(3*(omega * tau)**
           (2 * alpha) - (omega * tau)**2))/ \
        (2 * (omega * tau)**3 * np.sin(alpha * np.pi / 2) +
         (omega * tau)**(3 * alpha) + (omega * tau)**(alpha + 2))**2

def dv_eprime(v, e_prime, e_2prime):
    """Returns the derivative of the propagation speed with respect to
    e`

    -------

    """
    # mu_0 = __VAC_PERMEABILITY
    # eps_0 = __VAC_PERMITTIVITY
    return - 1/v * 1/(2 * e_prime * np.sqrt(1 + (e_2prime / e_prime)**2))

def dv_e2prime(v, e_prime, e_2prime):
    """Returns the derivative of the propagation speed with respect to
    e``

    -------

    """
    mu_0 = __VAC_PERMEABILITY
    eps_0 = __VAC_PERMITTIVITY
    return (- mu_0 * eps_0/2 * e_2prime / e_prime
            * 1/np.sqrt(1 + (e_2prime / e_prime)**2) * 1/v**3)


def deps_prime(freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s, d_tau, d_alpha,
               d_shift):
    """Returns the differential of the real part of the permittivity
    """
    deps = np.sqrt(np.abs(d_eprime_e_s(freqs=freqs, tau=tau, alpha=alpha))**2
                   * d_e_s**2 +\
           np.abs(d_eprime_e_h(freqs=freqs, tau=tau, alpha=alpha))**2 *
                   d_e_h**2 +\
           np.abs(d_eprime_tau(freqs=freqs, e_h=e_h, e_s=e_s, tau=tau,
                            alpha=alpha))**2 * d_tau**2 +\
           np.abs(d_eprime_alpha(freqs=freqs, e_h=e_h, e_s=e_s, tau=tau,
                            alpha=alpha))**2 * d_alpha**2 + d_shift**2)
    return deps


def deps_2prime(freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s, d_tau, d_alpha,
               d_shift):
    """Returns the differential of the imaginary part of the permittivity
    """
    deps = np.sqrt(
           np.abs(d_e2prime_e_s(freqs=freqs, tau=tau, alpha=alpha))**2 *
           d_e_s**2 + \
            np.abs(d_e2prime_e_h(freqs=freqs, tau=tau, alpha=alpha))**2 *
           d_e_h**2 + \
            np.abs(d_e2prime_tau(freqs=freqs, e_h=e_h, e_s=e_s, tau=tau,
                            alpha=alpha))**2 * d_tau**2 + \
            np.abs(d_e2prime_alpha(freqs=freqs, e_h=e_h, e_s=e_s, tau=tau,
                             alpha=alpha))**2 * d_alpha**2 + d_shift**2)
    return deps


def dv(v, e_prime, e_2prime, freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s,
       d_tau, d_alpha, d_shift):
    """Returns the differential of the propagation speed
    """
    _dv = np.sqrt(
           np.abs(dv_eprime(v=v, e_prime=e_prime, e_2prime=e_2prime))**2 *
           np.abs(deps_prime(freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s, d_tau,
                       d_alpha, d_shift))**2 +
           np.abs(dv_e2prime(v=v, e_prime=e_prime, e_2prime=e_2prime))**2 *
           np.abs(deps_2prime(freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s,
                    d_tau, d_alpha, d_shift))**2)
    return _dv

def dphi(v, e_prime, e_2prime, freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s,
       d_tau, d_alpha, d_shift, L, dL=0.000005):
    """Returns the differential of the phase

    """
    dv_ = dv(v, e_prime, e_2prime, freqs, e_h, e_s, tau, alpha, d_e_h, d_e_s,
       d_tau, d_alpha, d_shift)

    dphi_ = np.sqrt((2 * np.pi * freqs * L / v**2 * dv_)**2 +
                    (2 * np.pi * freqs / v * dL)**2)

    return dphi_

def dv_avg(v, phi,  e_prime, e_2prime, freqs, e_h, e_s, tau, alpha, d_e_h,
           d_e_s, d_tau, d_alpha, d_shift, L, dL=0.000005):
    """Returns the differential of the average propagation speed
    extracted from the phase

    """
    # Uncertainty in the phase
    dphi_ = dphi(v, e_prime, e_2prime, freqs, e_h, e_s, tau, alpha, d_e_h,
                 d_e_s, d_tau, d_alpha, d_shift, L, dL=0.000005)

    dv_avg_ = np.sqrt(np.abs(2 * np.pi * freqs / phi)**2 * dL**2
                      + np.abs(2 * np.pi * L / phi**2)**2 * dphi_**2)

    return dv_avg_