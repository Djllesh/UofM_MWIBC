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

    Returns
    -------

    """
    omega = 2 * np.pi * freqs
    return ((1 + (omega * tau)**(1 - alpha)*np.sin(alpha * np.pi /2))
            (1 + 2*(omega * tau)**(1-alpha)*np.sin(alpha * np.pi/2) +
              (omega * tau)**(2*(1 - alpha))))


def d_eprime_e_h(freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to e_h

    Returns
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

    Returns
    -------

    """
    omega = 2 * np.pi * freqs
    return ((e_h - e_s)*(omega* tau)**(alpha + 2)*(np.pi * tau * omega *
            np.cos(alpha * np.pi / 2)(omega**2 * tau**2 - (omega * tau)**
            (2*alpha)) + 2 * np.log(omega * tau)*(omega * tau *
            np.sin(alpha * np.pi / 2)*(3 * (omega * tau)**(2 * alpha) - (
            omega * tau)**2) + 2 * (omega * tau)**(3 * alpha)))) /\
            (2 * (2 * omega**3 * tau**3*np.sin(alpha * np.pi / 2) +
            (omega * tau)**(3 * alpha) + (omega * tau)**(alpha + 2))**2)


def d_eprime_tau(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the real part of the complex permittivity
    with respect to tau

    Returns
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

    Returns
    -------

    """
    omega = 2 * np.pi * freqs
    return (((omega * tau)**(1 - alpha)*np.cos(alpha * np.pi /2))
            (1 + 2*(omega * tau)**(1-alpha)*np.sin(alpha * np.pi/2) +
             (omega * tau)**(2*(1 - alpha))))


def d_e2prime_e_h(freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to e_h

    Returns
    -------

    """
    omega = 2 * np.pi * freqs
    return  - (((omega * tau) ** (1 - alpha) * np.cos(alpha * np.pi / 2))
                 / (1 + 2 * (omega * tau) ** (1 - alpha) * np.sin(alpha * np.pi /
                  2) + (omega * tau) ** (2 * (1 - alpha))))


def d_e2prime_alpha(e_h, e_s, freqs, tau, alpha):
    """Uncertainty in the imaginary part of the complex permittivity
    with respect to alpha

    Returns
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

    Returns
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

    Returns
    -------

    """
    mu_0 = __VAC_PERMEABILITY
    eps_0 = __VAC_PERMITTIVITY
    return - 1 / 4 * v**3 * (mu_0 * eps_0 * ((np.sqrt(1 + (e_prime /
            e_2prime)**2) + 1) + (e_prime / e_2prime)**2*(np.sqrt(1 + (
        e_prime / e_2prime)**2))**(-1)))

def dv_e2prime(v, e_prime, e_2prime):
    """Returns the derivative of the propagation speed with respect to
    e``

    Returns
    -------

    """
    mu_0 = __VAC_PERMEABILITY
    eps_0 = __VAC_PERMITTIVITY
    return  1 / 4 * mu_0 * eps_0 / 2 * (np.sqrt(1 + (e_prime /
             e_2prime)**2))**(-1)*e_prime**3/e_2prime**2

