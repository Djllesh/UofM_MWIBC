"""
Illia Prykhodko
University of Manitoba
November 25th, 2024
"""

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.loadsave import load_pickle
from umbms.beamform.propspeed import (cole_cole, phase_shape, phase_diff_MSE,
                                      phase_shape_wrapped)

__DATA_DIR = os.path.join(get_proj_path(),
                          'data/umbmid/cyl_phantom/speed_paper/')
__FIG_DIR = os.path.join(get_proj_path(),
                         'output/cyl_phantom/')
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
target_phase_unwrapped = np.unwrap(phase[1, :, 0])
length = 0.42

if __name__ == "__main__":

    # create a figure
    fig, ax = plt.subplots(1, 1, sharex=True, sharey='row',
                           **dict(figsize=(1200 / __MY_DPI, 800 / __MY_DPI),
                                  dpi=__MY_DPI))

    results = minimize(fun=phase_diff_MSE,
                       x0=np.array([3.40, 17.93, 101.75, 0.18,
                                    - target_phase[0]]),
                       # bounds=((1, 7), (8, 80), (7, 103),
                       #         (0.0, 0.25), (None, None)),
                       args=(target_phase, freqs, length, True),
                       method='trust-constr', options={'maxiter': 2000})
    print(results)
    e_h, e_s, tau, alpha, shift = results.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape = phase_shape_wrapped(freqs, length, epsilon, shift)

    ax.plot(freqs, target_phase, 'k--', label='Experimental phase',
            linewidth=0.7)

    ax.plot(freqs, shape, 'k-', label='Fit',
            linewidth=1.5)

    ax.set_ylim(-10, 10)
    ax.legend()
    ax.grid()
    ax.set_ylabel('Phase, (rad)')
    ax.set_xlabel('Frequency, (Hz)')
    ax.set_title('Fit to Unwrapped Phase')
    plt.tight_layout()
    fig.savefig(os.path.join(get_proj_path(),
                             'output/cyl_phantom/unwrapped_phase_ini.png'),
                transparent=True)