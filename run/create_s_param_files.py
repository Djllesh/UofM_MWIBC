from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import (load_pickle, save_pickle,
                            clean_birrs_folder, load_birrs_txt)

import os
import numpy as np

__DATA_DIR = 'C:/Users/illia/Desktop/Experimental data/20230821/20230821/'
# __DATA_DIR = 'C:/Users/illia/Desktop/MWIBC/UofM_MWIBC/output/' \
#              'cyl_phantom/recons/Immediate reference/Gen 2/' \
#              'Intensity comparison/intensity_dict.pickle'

# a = load_pickle(os.path.join(__DATA_DIR, 's11_data.pickle'))


clean_birrs_folder(__DATA_DIR)
fs_in_dir = os.listdir(__DATA_DIR)

s11_data = np.zeros([(int(len(fs_in_dir)/2)), 1001, 72],
                    dtype=complex)
s21_data = np.zeros_like(s11_data)

for file in fs_in_dir:

    if len(file) == 13:
        idx = int(file.split('_')[1][4])
    else:
        idx = int(file.split('_')[1][4:6])

    if 's11' in file:
        s11_scan_data = load_birrs_txt(os.path.join(__DATA_DIR, file))
        s11_data[idx, :, :] = s11_scan_data
    else:
        s21_scan_data = load_birrs_txt(os.path.join(__DATA_DIR, file))
        s21_data[idx, :, :] = s21_scan_data

save_pickle(s11_data, os.path.join(__DATA_DIR, 's11_data.pickle'))
save_pickle(s21_data, os.path.join(__DATA_DIR, 's21_data.pickle'))