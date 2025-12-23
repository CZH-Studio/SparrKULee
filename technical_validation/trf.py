from pathlib import Path

import torch
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


root_dir = Path(
    "/hdd/EEG/Audio/SparrKULee/derivatives/preprocessed/split/train/sub-001/ses-shortstories01_task-listeningActive_run-01"
)
eeg_name = "train_sub-001_ses-shortstories01_task-listeningActive_run-01_audiobook-5-1_eeg-512-low-gamma.pt"
env_name = "train_sub-001_ses-shortstories01_task-listeningActive_run-01_audiobook-5-1_envelope-512-board-band.pt"

eeg: NDArray = (
    torch.load(root_dir / eeg_name, weights_only=True).to(torch.float32).numpy()
)
env: NDArray = (
    torch.load(root_dir / env_name, weights_only=True)
    .to(torch.float32)
    .squeeze(1)
    .numpy()
)


def build_lag_matrix(env: NDArray, n_lags: int):
    """
    env: Tensor [T]
    return: X [T, n_lags]
    """
    T = env.shape[0]
    X = np.zeros((T, n_lags))
    for lag in range(n_lags):
        X[lag:, lag] = env[: T - lag]
    return X


def compute_trf(
    env,
    eeg,
    fs,
    tmin=0.0,
    tmax=0.3,
    lam=1e3,
):
    """
    env: Tensor [T]
    eeg: Tensor [T, C]
    fs: sampling rate (Hz)
    return:
        W: Tensor [C, n_lags]  (TRF for each channel)
    """
    T, C = eeg.shape

    n_lags = int((tmax - tmin) * fs)
    assert n_lags > 0, "n_lags must be > 0"

    # 构造延迟矩阵
    X: NDArray = build_lag_matrix(env, n_lags)

    # 去掉前 n_lags 个无效时间点
    X = X[n_lags:]
    eeg = eeg[n_lags:]

    # 岭回归通用部分
    Xt = X.T  # [n_lags, T']
    XtX = Xt @ X  # [n_lags, n_lags]
    reg = lam * np.eye(n_lags)

    W = np.zeros((C, n_lags))

    for c in range(C):
        y = eeg[:, c]  # [T']
        w = np.linalg.solve(XtX + reg, Xt @ y)
        W[c] = w

    return W


def plot_single_trf(W, ch_idx, fs):
    n_lags = W.shape[1]
    t = np.arange(n_lags) / fs * 1000  # ms

    plt.figure(figsize=(6, 3))
    plt.plot(t, W[ch_idx])
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("Latency (ms)")
    plt.ylabel("TRF weight")
    plt.title(f"Channel {ch_idx}")
    plt.tight_layout()
    plt.show()


def plot_region_trf(W: NDArray, region_channels, fs, label):
    n_lags = W.shape[1]
    t = np.arange(n_lags) / fs * 1000

    W_region = W[region_channels]
    mean = W_region.mean(axis=0)
    std = W_region.std(axis=0)

    plt.figure(figsize=(6, 3))
    plt.plot(t, mean, label=label)
    plt.fill_between(t, mean - std, mean + std, alpha=0.3)
    plt.xlabel("Latency (ms)")
    plt.ylabel("TRF weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"trf_{label}.png", dpi=300)


def bandpass_filter_eeg(
    eeg,
    fs,
    low,
    high,
    order=4,
):
    """
    eeg: ndarray [T, C]
    fs: sampling rate
    low, high: band edges (Hz)
    return: ndarray [T, C]
    """
    sos = butter(
        order,
        [low, high],
        btype="band",
        fs=fs,
        output="sos",
    )
    eeg_filt = sosfiltfilt(sos, eeg, axis=0)
    return eeg_filt


bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

band_name = "beta"
fs = 512
# eeg = bandpass_filter_eeg(eeg, fs, *bands[band_name])
tmin = 0.0
tmax = 0.3
lam = 1e3

W = compute_trf(
    env=env,
    eeg=eeg,
    fs=fs,
    tmin=tmin,
    tmax=tmax,
    lam=lam,
)
left_temporal = [
    7,  # FT7
    8,  # FC5
    13,  # C5
    14,  # T7
    15,  # TP7
    16,  # CP5
]

right_temporal = [
    42,  # FT8
    43,  # FC6
    50,  # C6
    51,  # T8
    52,  # TP8
    53,  # CP6
]

print("TRF shape:", W.shape)  # [64, n_lags]
plot_region_trf(W, left_temporal, fs, "Left Temporal")
plot_region_trf(W, right_temporal, fs, "Right Temporal")
