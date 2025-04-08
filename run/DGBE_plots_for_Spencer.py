import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

__DATA_DIR = "C:/Users/prikh/Desktop/Exp data/"

if __name__ == "__main__":
    df_pre = pd.read_csv(
        os.path.join(__DATA_DIR, "20231129/20231129_DGBE90.csv")
    )
    df_post = pd.read_csv(
        os.path.join(__DATA_DIR, "20240109/20240111_DGBE90.csv")
    )

    freqs = np.array(df_pre["Freqs"].values, dtype=float) * 1e6

    pre_spencer_perm = np.array(df_pre["Permittivity"].values)
    pre_spencer_cond = np.array(df_pre["Conductivity"].values)
    post_spencer_perm = np.array(df_post["Permittivity"].values)
    post_spencer_cond = np.array(df_post["Conductivity"].values)

    fig, ax = plt.subplots(1, 2, sharex=True)
    ax[0].plot(freqs, pre_spencer_perm, "b")
    ax[0].plot(freqs, post_spencer_perm, "r")
    ax[1].plot(freqs, pre_spencer_cond, "b")
    ax[1].plot(freqs, post_spencer_cond, "r")

    plt.show()
