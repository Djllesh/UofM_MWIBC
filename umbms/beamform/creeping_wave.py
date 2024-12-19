"""
Illia Prykodko

University of Manitoba
December 18th, 2024
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
target_phase_unwrapped = np.unwrap(phase[1, :, 0]) - 2 * 5 * np.pi
length = 0.42

creeping_phase = - np.pi * 2 * freqs * length / 3e8

if __name__ == "__main__":
    fig, ax = plt.subplots()

    ax.plot(freqs, target_phase_unwrapped, 'b-', linewidth=1,
            label='Experimental phase')
    ax.plot(freqs, creeping_phase, 'r-', linewidth=1.2,
            label='Creeping wave phase')
    ax.grid('-')
    ax.set_ylabel('Phase, rad')
    ax.set_xlabel('Frequencies, (Hz)')
    plt.tight_layout()
    plt.show()
