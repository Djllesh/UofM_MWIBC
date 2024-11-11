import matplotlib.pyplot as plt
import os
import numpy as np
from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle
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
for i in range(np.size(phase, axis=2)):

    fig, ax = plt.subplots(1, 2, sharex=True,
                           **dict(figsize=(1400/__MY_DPI, 1000/__MY_DPI),
                                           dpi=__MY_DPI))
    unwrapped_phase = np.unwrap(phase[1, :, i]) - 5 * 2 * np.pi
    propagation_speed = - 2 * np.pi * freqs * 0.42 / unwrapped_phase

    ax[0].plot(freqs, unwrapped_phase, 'k-', linewidth=1.5,
             label=r'Unwrapped target $(\varphi - 5 \pi)$')
    ax[0].plot(freqs, phase[1, :, i], 'b--', linewidth=1., label='Wrapped')
    ax[0].axhline(y=unwrapped_phase[0], linestyle='--', color='black',
                  linewidth=0.9)

    ax[0].set_xlabel("Frequency, (Hz)")
    ax[0].set_ylabel("Phase, rad")
    ax[0].set_title("$S_{21}$ phase")
    ax[0].legend()
    ax[0].grid('--')

    ax[1].grid('--')
    ax[1].plot(freqs, propagation_speed, 'r-', linewidth=1.5,
               label="$v(f) = - 2 \pi f L/ \\varphi(f)$")
    ax[1].set_xlabel("Frequency, (Hz)")
    ax[1].set_ylabel("Propagation speed (m/s)")
    ax[1].set_title("Frequency dependent propagation speed")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join('C://Users/prikh/Desktop/propagation_speed/',
                             'antenna_pos_%d.png') % i,
                dpi=__MY_DPI)
    plt.close(fig)