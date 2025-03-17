"""
Illia Prykhodko
University of Manitoba
December 13th, 2024
"""
import pandas
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
from umbms import get_proj_path
from umbms.loadsave import load_pickle
from umbms.beamform.propspeed import (cole_cole, phase_shape, phase_diff_MSE,
                                      phase_shape_explicit, fit_bootstrap,
                                      get_breast_speed_freq,
                                      fit_bootstrap_minimize,
                                      get_speed_from_epsilon)
from umbms.beamform.phase_speed_uncertainties import dv_avg, dv

__DATA_DIR = os.path.join(get_proj_path(),
                          'data/umbmid/cyl_phantom/speed_paper/')
__DIEL_DIR = os.path.join(get_proj_path(),
                          'data/freq_data/')
__FIG_DIR = os.path.join(get_proj_path(),
                         'output/cyl_phantom/')
__FD_NAME = '20240819_s21_data.pickle'
__DIEL_NAME = '20240813_DGBE90.csv'

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

if __name__ == "__main__":

    # Read .csv file of permittivity and conductivity values
    df = pandas.read_csv(os.path.join(__DIEL_DIR, __DIEL_NAME),
                         delimiter=';', decimal=',', skiprows=9)
    diel_data_arr = df.values

    perms = np.array(diel_data_arr[:, 1])
    conds = np.array(diel_data_arr[:, 3])

    p0 = [4., 44., 55., 0.125, 0.]

    # fit, error = fit_bootstrap(p0=p0, datax=freqs,
    #                            datay=target_phase_unwrapped,
    #                            function=phase_shape_explicit,
    #                            length=length)

    fit, error = fit_bootstrap_minimize(p0=p0, datax=freqs,
                               datay=target_phase_unwrapped,
                               function=phase_diff_MSE,
                               length=length, resfunc=phase_shape_explicit)

    # Extract the errors in the parameter fit
    err_e_h, err_e_s, err_tau, err_alpha, err_shift = error

    e_h, e_s, tau, alpha, shift = fit
    epsilon = cole_cole(freqs, e_h, e_s, tau, alpha)
    shape_unwrapped = phase_shape(freqs, length, epsilon, shift)

    # v = get_speed_from_epsilon(epsilon=epsilon)

    raw_speed_5pi = - 2 * np.pi * freqs * length / (target_phase_unwrapped -
                                                    2 * np.pi * 5)

    raw_speed_6pi = - 2 * np.pi * freqs * length / (target_phase_unwrapped -
                                                    2 * np.pi * 6)

    phase_speed_5pi = - 2 * np.pi * freqs * length / (shape_unwrapped - 2 *
                                                      np.pi * 5)

    phase_speed_5pi_in = (
            phantom_width / (length / phase_speed_5pi -
                             (length - phantom_width) / 3e8))

    phase_speed_6pi = - 2 * np.pi * freqs * length / (shape_unwrapped - 2 *
                                                      np.pi * 6)

    phase_speed_6pi_in = (
            phantom_width / (length / phase_speed_6pi -
                             (length - phantom_width) / 3e8))

    dv_avg_5_pi = dv_avg(v=phase_speed_5pi, phi=shape_unwrapped-2*5*np.pi,
                     e_prime=np.real(epsilon), e_2prime=np.abs(np.imag(epsilon)),
                     freqs=freqs, e_h=e_h, e_s=e_s, tau=tau, alpha=alpha,
                     d_e_h=err_e_h, d_e_s=err_e_s, d_tau=err_tau,
                     d_alpha=err_alpha, d_shift=err_shift, L=length)

    dv_avg_6_pi = dv_avg(v=phase_speed_6pi, phi=shape_unwrapped-2*6*np.pi,
                     e_prime=np.real(epsilon), e_2prime=np.abs(np.imag(epsilon)),
                     freqs=freqs, e_h=e_h, e_s=e_s, tau=tau, alpha=alpha,
                     d_e_h=err_e_h, d_e_s=err_e_s, d_tau=err_tau,
                     d_alpha=err_alpha, d_shift=err_shift, L=length)

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

    # ax.plot(freqs, phase_speed_5pi, 'r--', label=r'Extracted speed, shift = '
    #                                              r'$-2 \cdot 5\pi$',
    #         linewidth=1.2)
    # ax.plot(freqs, phase_speed_6pi, 'b--', label=r'Extracted speed, shift = '
    #                                              r'$-2 \cdot 6\pi$',
    #         linewidth=1.2)



    ax.plot(freqs, experimental_speed, 'k-', label='Experimental speed',
            linewidth=1.3)

    ax.plot(freqs, phase_speed_5pi_in, 'r-',
            label=r'Estimated speed inside, shift = $-2 \cdot 5\pi$',
            linewidth=1.3)
    ax.plot(freqs, phase_speed_6pi_in, 'b-',
            label=r'Estimated speed inside, shift = $-2 \cdot 6\pi$',
            linewidth=1.3)

    # ax.plot(freqs, speed_creeping, 'g-', label='Creeping wave')

    ax.grid()
    ax.legend(prop={'size': 8})
    ax.set_xlabel('Frequency, (Hz)')
    ax.set_ylabel('Propagation speed, (m/s)')
    plt.tight_layout()
    plt.show()
