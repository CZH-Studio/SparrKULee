from pathlib import Path
import argparse

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import torch

root_dir = Path("/hdd/EEG/Audio/SparrKULee/derivatives/preprocessed")
eeg_dir = root_dir / "eeg"
stimuli_dir = root_dir / "stimuli"
exec_dir = Path(__file__).parent
corr_dir = exec_dir / "trf_corr"
plot_dir = exec_dir / "trf_plot"
topo_dir = exec_dir / "trf_topo"
for x in [corr_dir, plot_dir, topo_dir]:
    x.mkdir(exist_ok=True, parents=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eeg", type=str)
    parser.add_argument("-s", "--stimuli", type=str)
    parser.add_argument("-b", "--backward", action="store_true")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=0)
    parser.add_argument("--lag_min_ms", type=int, default=-100)
    parser.add_argument("--lag_max_ms", type=int, default=400)
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    return args


def align(eeg: NDArray, envelope: NDArray):
    min_len = min(eeg.shape[0], envelope.shape[0])
    eeg = eeg[:min_len]
    envelope = envelope[:min_len]
    return eeg, envelope


def bandpass(data: NDArray, fs: int, low: int, high: int, order: int = 4):
    ndim = data.ndim
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, data, axis=0)
    if ndim == 1:
        filtered = filtered.squeeze()
    return filtered


def auto_load(path: Path) -> NDArray:
    if path.suffix == ".pt":
        data_t: torch.Tensor = torch.load(path, weights_only=True, map_location="cpu")
        data: NDArray = data_t.numpy()
    elif path.suffix == ".npy":
        data: NDArray = np.load(path)
    else:
        raise NotImplementedError()

    if data.ndim == 2:
        if data.shape[0] < data.shape[1]:  # [C, T]
            data = data.T  # [T, C]
        if data.shape[1] == 1:
            data = data.squeeze(-1)  # [T]
    return data


def build_lag_matrix(stim: NDArray, lag_min: int, lag_max: int):
    T = len(stim)
    lags = np.arange(lag_min, lag_max + 1)
    L = len(lags)
    X = np.zeros((T, L))
    for i, lag in enumerate(lags):
        if lag < 0:
            X[-lag:, i] = stim[: T + lag]
        elif lag > 0:
            X[: T - lag, i] = stim[lag:]

        else:
            X[:, i] = stim
    return X


def build_lag_matrix_multichannel(eeg: NDArray, lag_min: int, lag_max: int):
    """
    eeg: [T, C]
    return: [T, C * L]
    """

    T, C = eeg.shape
    lags = np.arange(lag_min, lag_max + 1)
    L = len(lags)
    X = np.zeros((T, C * L))
    for i, lag in enumerate(lags):
        start = i * C
        end = (i + 1) * C
        if lag < 0:
            X[-lag:, start:end] = eeg[: T + lag]
        elif lag > 0:
            X[: T - lag, start:end] = eeg[lag:]
        else:
            X[:, start:end] = eeg
    return X


def train_trf(
    eeg_data_list: list[NDArray],
    envelope_data_list: list[NDArray],
    fs,
    lag_min_ms=-100,
    lag_max_ms=400,
    alpha=1.0,
    backward=False,
):
    lag_min = int(lag_min_ms / 1000 * fs)
    lag_max = int(lag_max_ms / 1000 * fs)
    X_all = []
    Y_all = []
    for env, eeg in zip(envelope_data_list, eeg_data_list):
        if not backward:
            X = build_lag_matrix(env, lag_min, lag_max)
            Y = eeg
        else:
            X = build_lag_matrix_multichannel(eeg, lag_min, lag_max)
            Y = env.reshape(-1, 1)
        X_all.append(X)
        Y_all.append(Y)
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    model = Ridge(alpha, fit_intercept=False)
    model.fit(X_all, Y_all)
    return model, np.arange(lag_min, lag_max + 1)


def predict_trf(
    model: Ridge,
    envelope: NDArray,
    eeg: NDArray,
    lag_min: int,
    lag_max: int,
    backward=False,
) -> NDArray:
    """Calc predict corr using TRF model

    :param model: TRF model
    :type model: Ridge
    :param envelope: envelope data
    :type envelope: NDArray
    :param eeg: eeg data
    :type eeg: NDArray
    :param lag_min: lag min time
    :type lag_min: int
    :param lag_max: lag max time
    :type lag_max: int
    :param backward: forward or backward model, defaults to False
    :type backward: bool, optional
    :return: corr
    :rtype: NDArray
    """
    if backward:
        X_test = build_lag_matrix_multichannel(eeg, lag_min, lag_max)
        env_pred = model.predict(X_test).squeeze()
        r, _ = pearsonr(env_pred, envelope)
        return np.array(r)
    else:
        X_test = build_lag_matrix(envelope, lag_min, lag_max)
        EEG_pred = model.predict(X_test)
        corrs = []
        for c in range(eeg.shape[1]):
            r, _ = pearsonr(EEG_pred[:, c], eeg[:, c])
            corrs.append(r)
        return np.array(corrs)
