"""
Illia Prykhodko
University of Manitoba
June 6, 2023
"""

from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import save_pickle, load_birrs_txt

import os
import numpy as np

date = "big_dataset_merge_fix"

__DATA_DIR = "C:/Users/prikh/Desktop/Exp data/" + date + "/"

fs_in_dir = os.listdir(__DATA_DIR)

s21_data = np.zeros([(int(len(fs_in_dir)) // 2), 1001, 72], dtype=complex)

s11_data = np.zeros([(int(len(fs_in_dir)) // 2), 1001, 72], dtype=complex)

unnumbered_str_len = len("S11_expt.txt")

for file in fs_in_dir:
    if "S21" in file:
        # Find the length of the digit succeeding the S21_expt
        digit_length = len(file) - unnumbered_str_len
        # Determine the index
        idx = int(file.split(".")[0][-digit_length:])
        # Save in the appropriate slot
        s21_scan_data = load_birrs_txt(os.path.join(__DATA_DIR, file))
        s21_data[idx, :, :] = s21_scan_data

    if "S11" in file:
        # Find the length of the digit succeeding the S21_expt
        digit_length = len(file) - unnumbered_str_len
        # Determine the index
        idx = int(file.split(".")[0][-digit_length:])
        # Save in the appropriate slot
        s11_scan_data = load_birrs_txt(os.path.join(__DATA_DIR, file))
        s11_data[idx, :, :] = s11_scan_data

save_pickle(
    s11_data, os.path.join(__DATA_DIR, "s11_big_dataset_correction.pickle")
)

save_pickle(
    s21_data, os.path.join(__DATA_DIR, "s21_big_dataset_correction.pickle")
)
