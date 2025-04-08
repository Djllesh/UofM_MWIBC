"""
Illia Prykhodko
University of Manitoba
November 24, 2023
"""

import numpy as np
import pandas as pd
import os
from umbms.loadsave import save_pickle

__DATA_DIR = "C:/Users/prikh/Desktop/Exp data/20240819/"

df = pd.read_csv(
    os.path.join(__DATA_DIR, "20240819_small_ball.csv"), delimiter=","
)
size = len(df["id"].values)

tar_dict = {
    "n_expt": None,
    "id": None,
    "tum_diam": np.nan,
    "tum_shape": None,
    "tum_x": np.nan,
    "tum_y": np.nan,
    "adi_ref_id": np.nan,
    "adi_ref_id2": np.nan,
    "rod_ref_id": np.nan,
    "emp_ref_id": np.nan,
    "date": "20240819",
    "ant_rad": 21.0,
    "type": None,
}

md = []

for i in range(size):
    md.append(tar_dict.copy())


for i in range(size):
    for key in df.keys():
        if key == "tum_shape":
            if df[key][i] == "nan":
                md[i][key] = None
            else:
                md[i][key] = df[key][i]

        elif "ref" in key and ~np.isnan(df[key][i]):
            md[i][key] = int(df[key][i])
        else:
            md[i][key] = df[key][i]


save_pickle(md, os.path.join(__DATA_DIR, "20240819_metadata.pickle"))

