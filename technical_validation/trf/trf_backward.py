from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import pandas as pd


def align(eeg: NDArray, envelope: NDArray):
    min_len = min(eeg.shape[0], envelope.shape[0])
    eeg = eeg[:min_len]
    envelope = envelope[:min_len]
    return eeg, envelope


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


def train_backward_trf(
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
        X = build_lag_matrix_multichannel(eeg, lag_min, lag_max)
        Y = env.reshape(-1, 1)
        X_all.append(X)
        Y_all.append(Y)
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    model = Ridge(alpha, fit_intercept=False)
    model.fit(X_all, Y_all)
    return model, np.arange(lag_min, lag_max + 1)


def predict_backward_trf(
    model: Ridge,
    eeg: NDArray,
    envelope: NDArray,
    lag_min: int,
    lag_max: int,
):
    X_test = build_lag_matrix_multichannel(eeg, lag_min, lag_max)
    env_pred = model.predict(X_test).squeeze()
    r, _ = pearsonr(env_pred, envelope)
    return r


def main():
    root_dir = Path("/hdd/EEG/Audio/SparrKULee/derivatives")
    eeg_dir = root_dir / "official_preprocessed_eeg"
    stimulus_dir = root_dir / "official_preprocessed_stimuli"
    subject_results = pd.DataFrame(columns=["corr"])
    for subject_dir in eeg_dir.glob("sub-*"):
        subject = int(subject_dir.stem[4:])
        run_dir = next(subject_dir.iterdir())
        subject_data_pairs: list[tuple[Path, Path]] = []
        for eeg_file in run_dir.glob("*.npy"):
            stimuli_name = eeg_file.stem.split("desc-preproc-audio-")[1]
            if stimuli_name.endswith("_eeg"):
                stimuli_name = stimuli_name[:-4]
            envelope_file = stimulus_dir / f"{stimuli_name}_envelope.npy"
            if envelope_file.exists():
                subject_data_pairs.append((eeg_file, envelope_file))
        cross_validation_results = []
        for i in range(len(subject_data_pairs)):
            eeg_data_list: list[NDArray] = []
            envelope_data_list: list[NDArray] = []
            for j, (eeg_file, envelope_file) in enumerate(subject_data_pairs):
                if i != j:
                    eeg_data: NDArray = np.load(eeg_file).T
                    envelope_data: NDArray = np.load(envelope_file).squeeze(-1)
                    eeg_data, envelope_data = align(eeg_data, envelope_data)
                    eeg_data_list.append(eeg_data)
                    envelope_data_list.append(envelope_data)
            model, lags = train_backward_trf(
                eeg_data_list,
                envelope_data_list,
                FS,
                LAG_MIN_MS,
                LAG_MAX_MS,
                ALPHA,
            )
            test_eeg_data: NDArray = np.load(subject_data_pairs[i][0]).T
            test_envelope_data: NDArray = np.load(subject_data_pairs[i][1]).squeeze(-1)
            test_eeg_data, test_envelope_data = align(
                test_eeg_data,
                test_envelope_data,
            )
            r = predict_backward_trf(
                model,
                test_eeg_data,
                test_envelope_data,
                lags[0],
                lags[-1],
            )
            cross_validation_results.append(r)
        subject_result = np.mean(cross_validation_results)
        print(f"Mean backward correlation of subject {subject}: {subject_result:.6f}")
        subject_results.loc[subject] = [subject_result]
    subject_results.sort_index().to_csv(
        "trf_backward_corr.csv",
        index=True,
        index_label="subject",
    )


if __name__ == "__main__":
    FS = 64
    LAG_MIN_MS = -100
    LAG_MAX_MS = 400
    ALPHA = 1.0
    main()
