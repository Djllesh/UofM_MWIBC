"""
Tyson Reimer
University of Manitoba
August 18th, 2023
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from umbms import get_proj_path
from umbms.loadsave import load_pickle

from umbms.beamform.iczt import iczt

###############################################################################


def iczt_NEW(fd_sig, ini_f, fin_f, ini_t, fin_t, n_ts):
    """ICZT for FD to TD conversion of 1D signal

    Parameters
    ----------
    fd_sig : array_like
        FD signal to be converted
    ini_f : float
        Initial freq of the fd_sig, in [Hz]
    fin_f : float
        Final freq of the fd_sig, in [Hz]
    ini_t : float
        Initial time-point at which the output td_sig will be
        determined, [s]
    fin_t : float
        Initial time-point at which the output td_sig will be
        determined, [s]
    n_ts : int
        Number of points between ini_t and fin_t at which
        the output td_sig will be evaluated

    Returns
    -------
    td_sig : array_like
        The time-domain signal corresponding to fd_sig
    """

    n_fs = len(fd_sig)  # Number of freqs

    df = np.diff(np.linspace(ini_f, fin_f, n_fs))[0]  # Freq step size
    dt = np.diff(np.linspace(ini_t, fin_t, n_ts))[0]  # Time step size

    f_indices = np.arange(n_fs)  # Freq-indices, k = 0, 1, ...
    t_indices = np.arange(n_ts)  # Time-indices, n = 0, 1, ...

    # Calculate the time domain signal via the ICZT
    td_sig = np.sum(fd_sig[None, :]
                    * np.exp(1j * 2 * np.pi * ini_t * df * f_indices)[None, :]
                    * np.power((np.exp(1j * 2 * np.pi * dt
                                       * (ini_f + df * f_indices)))[None, :],
                               t_indices[:, None]),
                    axis=1) / n_fs

    # Apply phase compensation
    td_sig *= np.exp(1j * 2 * np.pi * ini_f * ini_t)

    return td_sig


def czt(td_sig, ini_t, fin_t, ini_f, fin_f, n_fs):
    """CZT for TD to FD conversion of 1D signal

    Parameters
    ----------
    td_sig : array_like
        TD signal to be converted
    ini_t : float
        Initial time-point of the signal [s]
    fin_t : float
        Final time-point of the signal [s]
    ini_f : float
        Initial frequency at which the output fd_sig will be
        determined, [Hz]
    fin_f : float
        Final frequency at which the output fd_sig will be
        determined, [Hz]
    n_fs : int
        Number of points between ini_f and fin_f at which
        the output fd_sig will be evaluated

    Returns
    -------
    fd_sig : array_like
        The frequency-domain signal corresponding to td_sig
    """
    n_ts = len(td_sig)  # Number of freqs

    df = np.diff(np.linspace(ini_f, fin_f, n_fs))[0]  # Freq step size
    dt = np.diff(np.linspace(ini_t, fin_t, n_ts))[0]  # Time step size

    f_indices = np.arange(n_fs)  # Freq-indices, k = 0, 1, ...
    t_indices = np.arange(n_ts)  # Time-indices, n = 0, 1, ...

    # Calculate the FD signal
    fd_sig = np.sum(td_sig[None, :]
                    * np.exp(-1j * 2 * np.pi * ini_f * dt * t_indices)[None, :]
                    * np.power((np.exp(-1j * 2 * np.pi * df
                                       * (ini_t + dt * t_indices)))[None, :],
                               f_indices[:, None]),
                    axis=1)

    # Apply phase compensation
    fd_sig *= np.exp(-1j * 2 * np.pi * ini_f * ini_t)

    fd_sig *= (n_fs / n_ts)  # Scaling factor

    return fd_sig


###############################################################################

if __name__ == "__main__":

    fd = load_pickle(os.path.join(get_proj_path(),
                                  'data/umbmid/g3/g3_s11.pickle'))

    tar_sig = fd[0, :, 0]  # Target signal

    n_fs = len(tar_sig)  # Number of frequencies
    ini_f = 1e9  # Initial freq [Hz]
    fin_f = 9e9  # Final freq [Hz]
    freqs = np.linspace(ini_f, fin_f, n_fs)  # Scan frequencies
    df = np.diff(freqs)[0]  # Frequency step size

    # The time-points obtained when the IFFT is calculated
    ifft_ts = np.fft.fftfreq(n=n_fs, d=df)

    # Time-step when IFFT is calculated
    dt = np.diff(ifft_ts)[0]

    # The ICZT parameters required to ensure no info on the FD signal
    # is lost during the FD-to-TD conversion
    ini_t = np.min(ifft_ts)
    fin_t = np.max(ifft_ts)
    n_ts = n_fs * 8  # Can be any integer multiple of 1001

    # Calculate the TD signal using the ICZT as defined in this file
    td = iczt_NEW(fd_sig=tar_sig, ini_f=ini_f, fin_f=fin_f,
                  ini_t=ini_t, fin_t=fin_t, n_ts=n_ts)

    # Calculate the TD signal using the *old* implementation of
    # the ICZT, to make sure the ICZT defined in this file is correct
    td_old = iczt(fd_data=tar_sig, ini_f=ini_f, fin_f=fin_f,
                  ini_t=ini_t, fin_t=fin_t, n_time_pts=n_ts)

    # Recover the FD signal using the CZT as defined in this file
    rec_fd = czt(td_sig=td, ini_t=ini_t, fin_t=fin_t,
                 ini_f=ini_f, fin_f=fin_f, n_fs=n_fs)

    # Is the new ICZT the same as the old?
    plt.figure()
    plt.plot(td, 'k-', label='new')
    plt.plot(td_old, 'r--', label='old')
    plt.legend()
    plt.show()

    # Can we recover the FD signal?
    plt.figure()
    plt.plot(freqs, np.abs(tar_sig), 'k-', label='original')
    plt.plot(freqs, np.abs(rec_fd), 'r--', label='recovered')
    plt.legend()
    plt.show()