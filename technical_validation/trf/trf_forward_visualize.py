from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import mne

from datautils import CHANNELS


def align(eeg: NDArray, envelope: NDArray):
    min_len = min(eeg.shape[0], envelope.shape[0])
    eeg = eeg[:min_len]
    envelope = envelope[:min_len]
    return eeg, envelope


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


def train_trf(
    eeg_data_list: list[NDArray],
    envelope_data_list: list[NDArray],
    fs,
    lag_min_ms=-100,
    lag_max_ms=400,
    alpha=1.0,
):
    lag_min = int(lag_min_ms / 1000 * fs)
    lag_max = int(lag_max_ms / 1000 * fs)
    X_all = []
    Y_all = []
    for env, eeg in zip(envelope_data_list, eeg_data_list):
        X = build_lag_matrix(env, lag_min, lag_max)
        X_all.append(X)
        Y_all.append(eeg)
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    model = Ridge(alpha, fit_intercept=False)
    model.fit(X_all, Y_all)
    return model, np.arange(lag_min, lag_max + 1)


def plot_trf(weights: NDArray, lags, fs, subject: int, save_dir: Path):
    lag_ms = lags / fs * 1000
    mean_trf = weights.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(lag_ms, mean_trf)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Lag (ms)")
    plt.ylabel("TRF weight")
    plt.title(f"TRF - Subject {subject}")
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f"sub-{subject}_trf.png")
    plt.close()


def plot_trf_topography(weights: NDArray, lags, fs, channels, subject, save_dir):
    lag_ms = lags / fs * 1000
    target_times = [100, 200]
    indices = [np.argmin(np.abs(lag_ms - t)) for t in target_times]
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(channels, sfreq=fs, ch_types="eeg")
    info.set_montage(montage)
    fig, axes = plt.subplots(1, len(indices), figsize=(8, 4))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx, t in zip(axes, indices, target_times):
        data = weights[idx]
        mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            contours=0,
        )
        ax.set_title(f"{t} ms")
    plt.suptitle(f"TRF Topography - Subject {subject}")
    plt.tight_layout()
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / f"sub-{subject}_topography.png")
    plt.close()


def main():
    root_dir = Path("/hdd/EEG/Audio/SparrKULee/derivatives")
    eeg_dir = root_dir / "official_preprocessed_eeg"
    stimulus_dir = root_dir / "official_preprocessed_stimuli"
    trf_plot_dir = Path(__file__).parent / "trf_plot"
    trf_topo_dir = Path(__file__).parent / "trf_topo"
    for subject_dir in eeg_dir.glob("sub-*"):
        subject = int(subject_dir.stem[4:])
        run_dir = next(subject_dir.iterdir())
        subject_data_pairs: list[tuple[Path, Path]] = (
            []
        )  # eeg-envelope pairs of one subject
        for eeg_file in run_dir.glob("*.npy"):
            # parse stimuli name
            stimuli_name = eeg_file.stem.split("desc-preproc-audio-")[1]
            if stimuli_name.endswith("_eeg"):
                stimuli_name = stimuli_name[:-4]  # audiobook_1
            # find stimuli file, for trf, use envelope
            envelope_file = stimulus_dir / f"{stimuli_name}_envelope.npy"
            if envelope_file.exists():
                subject_data_pairs.append((eeg_file, envelope_file))
        # for visualize, use all data
        # get train data
        eeg_data_list: list[NDArray] = []
        envelope_data_list: list[NDArray] = []
        for eeg_file, envelope_file in subject_data_pairs:
            eeg_data: NDArray = np.load(eeg_file).T  # [t, C]
            envelope_data: NDArray = np.load(envelope_file).squeeze(-1)  # [t]
            eeg_data, envelope_data = align(eeg_data, envelope_data)
            eeg_data_list.append(eeg_data)
            envelope_data_list.append(envelope_data)
        # train trf
        model, lags = train_trf(
            eeg_data_list, envelope_data_list, FS, LAG_MIN_MS, LAG_MAX_MS, ALPHA
        )
        # get weight
        weights = model.coef_.T  # [lag, channel]
        plot_trf(weights, lags, FS, subject, trf_plot_dir)
        plot_trf_topography(weights, lags, FS, CHANNELS, subject, trf_topo_dir)


if __name__ == "__main__":
    FS = 64
    LAG_MIN_MS = -100
    LAG_MAX_MS = 400
    ALPHA = 1.0
    main()
