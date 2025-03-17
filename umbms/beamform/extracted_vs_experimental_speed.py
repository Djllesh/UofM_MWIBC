"""
Illia Prykhodko
University of Manitoba
November 26th, 2024
"""
import pandas
from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.loadsave import load_pickle
from umbms.beamform.propspeed import (cole_cole, phase_shape, phase_diff_MSE,
                                      phase_shape_wrapped,
                                      get_breast_speed_freq)
from umbms.analysis.stats import ccc

#
# __DATA_DIR = os.path.join(get_proj_path(),
#                           'data/umbmid/cyl_phantom/speed_paper/')
# __DIEL_DIR = os.path.join(get_proj_path(),
#                           'data/freq_data/')
# __FIG_DIR = os.path.join(get_proj_path(),
#                          'output/cyl_phantom/')
# __FD_NAME = '20240819_s21_data.pickle'
# __DIEL_NAME = '20240813_DGBE90.csv'

__DATA_DIR = os.path.join(get_proj_path(),
                          'data/umbmid/cyl_phantom/ursi_data/')
__DIEL_DIR = os.path.join(get_proj_path(),
                          'data/freq_data/')
__FIG_DIR = os.path.join(get_proj_path(),
                         'output/cyl_phantom/')

__FD_NAME = '20250115_s21_data.pickle'
# __FD_NAME = '20241219_s21_data.pickle'

__DIEL_NAME = '20250115_DGBE70.csv'

# __DIEL_NAME = '20241219_DGBE90.csv'

# __DIEL_NAME = '20241219_DGBE95.csv'

# __DIEL_NAME = '20241219_glycerin.csv'

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
phantom_width = 0.11
plt.plot(freqs, target_phase, 'r--')
plt.plot(freqs, target_phase_unwrapped, 'r-')
plt.show()
plt.close()
if __name__ == "__main__":

    # Read .csv file of permittivity and conductivity values
    df = pandas.read_csv(os.path.join(__DIEL_DIR, __DIEL_NAME),
                         delimiter=';', decimal=',', skiprows=9)
    diel_data_arr = df.values

    perms = np.array(diel_data_arr[:, 1])
    conds = np.array(diel_data_arr[:, 3])

    results = minimize(fun=phase_diff_MSE,
                       x0=np.array([3.40, 17.93, 101.75, 0.18,
                                    - target_phase[0]]),
                       bounds=((1, 7), (8, 80), (7, 103),
                               (0.0, 0.25), (None, None)),
                       args=(target_phase_unwrapped, freqs, length, False),
                       method='trust-constr', options={'maxiter': 2000})

    results_unwrapped = minimize(fun=phase_diff_MSE,
                                 x0=np.array(results.x),
                                 # bounds=((1, 7), (8, 80), (7, 103),
                                 #         (0.0, 0.25), (None, None)),
                                 args=(target_phase_unwrapped, freqs, length,
                                       False),
                                 method='trust-constr', tol=1e-15)

    e_h, e_s, tau, alpha, shift = results_unwrapped.x
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    raw_speed_5pi = - 2 * np.pi * freqs * length / (target_phase_unwrapped -
                                                    2 * np.pi * 5)

    raw_speed_6pi = - 2 * np.pi * freqs * length / (target_phase_unwrapped -
                                                    2 * np.pi * 6)

    phase_speed_5pi = - 2 * np.pi * freqs * length / (shape_unwrapped - 2 *
                                                      np.pi * 5)

    second_average = - 4 * np.pi * freqs * \
    np.sqrt(((length - phantom_width)/2 + phantom_width/2)**2 + (
            phantom_width/2)**2) / ( shape_unwrapped - 2 * np.pi * 6)

    phase_speed_5pi_in = (
            phantom_width / (length / phase_speed_5pi -
                              (length - phantom_width) / 3e8))

    phase_speed_6pi = - 2 * np.pi * freqs * length / (shape_unwrapped - 2 *
                                                      np.pi * 6)

    phase_speed_6pi_in = (
            phantom_width / (length / phase_speed_6pi -
                             (length - phantom_width) / 3e8))

    # speed_creeping = phantom_width / (length / second_average -
    #                      (length - phantom_width) / 3e8)

    experimental_speed = get_breast_speed_freq(freqs=freqs,
                                               permittivities=perms,
                                               conductivities=conds)
    __MY_DPI = 120
    fig, ax = plt.subplots(**dict(figsize=(1500 / __MY_DPI, 720 / __MY_DPI),
                                          dpi=__MY_DPI))

    ax.plot(freqs, raw_speed_5pi, 'r--', label=r'Unfitted speed '
                                               r'$-2 \cdot 5\pi$',
            linewidth=0.7)
    ax.plot(freqs, raw_speed_6pi, 'b--', label=r'Unfitted speed '
                                               r'$-2 \cdot 6\pi$',
            linewidth=0.7)

    ax.plot(freqs, phase_speed_5pi, 'r--', label=r'Extracted speed, shift = '
                                                r'$-2 \cdot 5\pi$',
            linewidth=1.2)
    ax.plot(freqs, phase_speed_6pi, 'b--', label=r'Extracted speed, shift = '
                                                r'$-2 \cdot 6\pi$',
            linewidth=1.2)

    ax.plot(freqs, experimental_speed, 'k-', label='Experimental speed',
            linewidth=1.3)

    ax.plot(freqs, phase_speed_5pi_in, 'r-',
            label=r'Estimated speed inside, shift = $-2 \cdot 5\pi$',
            linewidth=1.3)
    ax.plot(freqs, phase_speed_6pi_in, 'b-',
            label=r'Estimated speed inside, shift = $-2 \cdot 6\pi$',
            linewidth=1.3)

    # ax.plot(freqs, speed_creeping, 'g-', label='Creeping wave')

    cor = pearsonr(phase_speed_6pi_in, experimental_speed)[0]

    print(ccc(phase_speed_6pi_in, experimental_speed))

    ax.grid()
    ax.legend(prop={'size': 8})
    ax.set_xlabel('Frequency, (Hz)')
    ax.set_ylabel('Propagation speed, (m/s)')
    plt.tight_layout()
    plt.show()
