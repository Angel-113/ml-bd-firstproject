import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from multiprocessing import Pool

def read_file(path: str, bd: bool = True) -> pd.DataFrame:
    data_frame = pd.read_csv(path, header=None)
    if bd:
        data_frame["Activity"] = path.split("/")[-3]
    return data_frame

def read_bd(data_path: str) -> pd.DataFrame:
    files_path = sorted(glob.glob(os.path.join(data_path, '**/*.txt'), recursive=True))
    with Pool(processes=4) as pool:
        data_frame = pd.concat(pool.map(read_file, files_path))
        data_frame.to_csv("../AnalisisDatos/data/analisis.csv", index=False)
        return data_frame

def check_bd(path: str) -> bool:
    return os.path.exists(path)

def slicing(df: pd.DataFrame, activity: int, sensors: int) -> pd.DataFrame:
    activity = 1 if activity < 1 else activity % 20
    sensors = 1 if sensors < 1 else sensors % 6
    return df.iloc[(60000 * (activity - 1)) + 1: (60000 * activity) + 1, 9 * (sensors - 1): 9 * sensors]

def attrb(df: pd.DataFrame, sensors: int) -> pd.DataFrame:
    sensors = 1 if sensors < 1 else sensors % 6
    return df.iloc[0: 1140000, 9 * (sensors - 1): 9 * sensors]

def slice_by_activity(df: pd.DataFrame, activity: int) -> pd.DataFrame:
    return df.iloc[60000 * activity: 60000 * (activity + 1), 0: 47]

def draw_corr_matr(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coeficiente de correlación", rotation=270, labelpad=20)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(corr.columns, fontsize=14)

    ax.set_xticks(np.arange(len(corr.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr.columns)) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    plt.title("Matríz de correlación", fontsize=16, pad=20)
    plt.xlabel('Atributos', fontsize=16)
    plt.ylabel('Atributos', fontsize=16)
    plt.tight_layout()

    plt.show()
