"""
Illia Prykhodko
University of Manitoba
November 11th, 2024
"""

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.loadsave import load_pickle
from umbms.beamform.propspeed import cole_cole, phase_shape, phase_diff_MSE

__DATA_DIR = os.path.join(get_proj_path(),
                          'data/umbmid/cyl_phantom/speed_paper/')

__FD_NAME = '20240819_s21_data.pickle'

# the frequency parameters from the scan
__INI_F = 2e9
__FIN_F = 9e9
__N_FS = 1001
__MY_DPI = 120

data = load_pickle(os.path.join(__DATA_DIR, __FD_NAME))
freqs = np.linspace(__INI_F, __FIN_F, __N_FS)
phase = np.angle(data)
target_phase = phase[1, :, 0]
target_phase_unwrapped5 = np.unwrap(target_phase) - 5 * 2 * np.pi
target_phase_unwrapped6 = np.unwrap(target_phase) - 6 * 2 * np.pi
length = 0.42

if __name__ == "__main__":

    # Create a figure
    fig, ax = plt.subplots(2, 2, sharex=True, sharey='row',
                           **dict(figsize=(1920 / __MY_DPI, 1080 / __MY_DPI),
                                  dpi=__MY_DPI))

    # Create a mask for extrapolation
    mask = freqs > 3.5e9

    # List of arguments for the fit
    args = [(target_phase_unwrapped6, freqs, length),
            (target_phase_unwrapped5, freqs, length),
            (target_phase_unwrapped6[mask], freqs[mask], length),
            (target_phase_unwrapped5[mask], freqs[mask], length)]

    # List of fit results
    results = [minimize(fun=phase_diff_MSE,
                        x0=np.array([3.40, 17.93, 101.75, 0.18, 10 * np.pi]),
                        bounds=((1, 7), (8, 80), (7, 103),
                              (0.0, 0.25), (None, None)),
                        args=arg,
                        method='Nelder-Mead', tol=1e-5) for arg in args]

    # Top row - phase
    for idx, axis in enumerate(ax[0]):
        e_h6, e_s6, tau6, alpha6, shift6 = results[2 * idx].x
        e_h5, e_s5, tau5, alpha5, shift5 = results[2 * idx + 1].x

        exp_vals6 = args[0][0]
        exp_vals5 = args[1][0]

        extracted_shape6 = phase_shape(freqs, length,
                                  epsilon=cole_cole(freqs, e_h6, e_s6, tau6,
                                                    alpha6), shift=shift6)
        extracted_shape5 = phase_shape(freqs, length,
                                       epsilon=cole_cole(freqs, e_h5, e_s5, tau5,
                                                         alpha5), shift=shift5)

        axis.plot(freqs, target_phase_unwrapped6, 'k-', linewidth=1.5,
                 label=r'Experimental phase $6\cdot 2pi$ shift')
        axis.plot(freqs, target_phase_unwrapped5, 'r-', linewidth=1.5,
                 label=r'Experimental phase $5\cdot 2pi$ shift')
        axis.plot(freqs, extracted_shape6, 'k--', linewidth=0.7,
                 label=r'Extracted shape $6\cdot 2pi$ shift')
        axis.plot(freqs, extracted_shape5, 'r--', linewidth=0.7,
                 label=r'Extracted shape $5\cdot 2pi$ shift')

    # Bottom row - speed
    for idx, axis in enumerate(ax[1]):

        e_h6, e_s6, tau6, alpha6, shift6 = results[2 * idx].x
        e_h5, e_s5, tau5, alpha5, shift5 = results[2 * idx + 1].x

        exp_vals6 = args[0][0]
        exp_vals5 = args[1][0]

        extracted_shape6 = phase_shape(freqs, length,
                                       epsilon=cole_cole(freqs, e_h6, e_s6, tau6,
                                                         alpha6), shift=shift6)
        extracted_shape5 = phase_shape(freqs, length,
                                       epsilon=cole_cole(freqs, e_h5, e_s5, tau5,
                                                         alpha5), shift=shift5)

        experimental_speed5 = (-2 * np.pi * length * freqs /
                               exp_vals5)

        experimental_speed6 = (-2 * np.pi * length * freqs /
                               exp_vals6)

        axis.plot(freqs, experimental_speed6,
                      'k-', linewidth=1.5, label='Experimental speed $6\cdot 2pi$ shift')
        axis.plot(freqs, -2 * np.pi * length * freqs / extracted_shape6,
                      'k--', linewidth=.7, label='Theoretical speed $6\cdot 2pi$ shift')
        axis.plot(freqs, experimental_speed5,
                      'r-', linewidth=1.5, label='Experimental speed $5\cdot 2pi$ shift')
        axis.plot(freqs, -2 * np.pi * length * freqs / extracted_shape5,
                      'r--', linewidth=.7, label='Theoretical speed $5\cdot 2pi$ shift')

    plt.xlabel('Frequency, (Hz)')
    ax[0, 0].set_ylabel('Phase, (rad)')
    ax[1, 0].set_ylabel('Propagation speed, (m/s)')
    ax[0, 0].set_title('Full fit')
    ax[0, 1].set_title('Partial fit (>3.5 GHz)')

    # Create the grid and the legend on every subplot
    for axis in ax.flat:
        axis.grid('--')
        axis.legend()

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(get_proj_path(),
                             'output/cyl_phantom/unwrap_speed.png'),
                transparent=True)
