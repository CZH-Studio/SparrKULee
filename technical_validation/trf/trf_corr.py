from pathlib import Path


import numpy as np
from numpy.typing import NDArray
import pandas as pd


from .utils import (
    align,
    train_trf,
    predict_trf,
    auto_load,
    parse_args,
    bandpass,
    corr_dir,
    eeg_dir,
    stimuli_dir,
)


def main():
    args = parse_args()
    eeg_feature: str = args.eeg
    stimuli_feature: str = args.stimuli
    fs: int = args.fs
    low: int = args.low
    high: int = args.high
    lag_min_ms: int = args.lag_min_ms
    lag_max_ms: int = args.lag_max_ms
    shift: int = args.shift
    alpha: float = args.alpha
    backward: bool = args.backward

    direction = "backward" if backward else "forward"
    lags = f"[{lag_min_ms},{lag_max_ms}]"
    window = f"[{low},{high}]" if high > low else "bandwidth"
    shift_str = f"_shift={shift}" if shift != 0 else ""

    # init final results, iter for each subject
    subject_results = pd.DataFrame(columns=["corr"])
    for subject_dir in eeg_dir.glob("sub-*"):
        subject = int(subject_dir.stem[4:])  # subject id
        # eeg-envelope pairs (filepath) of one subject
        subject_data_pairs: list[tuple[Path, Path]] = []
        # find all eeg-envelope pairs of one subject
        for run_dir in subject_dir.iterdir():
            for eeg_file in run_dir.iterdir():
                _, _, _, _, stimuli_name, eeg_feature_name = eeg_file.stem.split("_")
                if eeg_feature_name == eeg_feature:
                    stimuli_file = (
                        stimuli_dir
                        / stimuli_name
                        / f"{stimuli_name}_{stimuli_feature}.pt"
                    )
                    subject_data_pairs.append((eeg_file, stimuli_file))
                    break

        # use cross-validation result
        cross_validation_results = []
        for i in range(len(subject_data_pairs)):
            # get train data
            eeg_data_list: list[NDArray] = []
            envelope_data_list: list[NDArray] = []
            for j, (eeg_file, envelope_file) in enumerate(subject_data_pairs):
                if i != j:
                    eeg_data = auto_load(eeg_file)
                    envelope_data = auto_load(envelope_file)
                    eeg_data, envelope_data = align(eeg_data, envelope_data)
                    # apply bandpass filter if need
                    if high > low:
                        eeg_data = bandpass(eeg_data, fs, low, high)
                        envelope_data = bandpass(envelope_data, fs, low, high)
                    # shift stimuli to verify the corr is valid
                    if shift > 0:
                        n_shift = shift * fs
                        envelope_data = np.roll(envelope_data, n_shift)
                    eeg_data_list.append(eeg_data)
                    envelope_data_list.append(envelope_data)
            # train trf
            model, lags = train_trf(
                eeg_data_list,
                envelope_data_list,
                fs,
                lag_min_ms,
                lag_max_ms,
                alpha,
                backward,
            )
            # predict on test data
            test_eeg_data = auto_load(subject_data_pairs[i][0])
            test_envelope_data = auto_load(subject_data_pairs[i][1])
            test_eeg_data, test_envelope_data = align(test_eeg_data, test_envelope_data)
            corrs = predict_trf(
                model, test_envelope_data, test_eeg_data, lags[0], lags[-1], backward
            )
            cross_validation_results.append(np.mean(corrs))
        subject_result = np.mean(cross_validation_results)
        print(
            f"Mean {direction} correlation of subject {subject}: {subject_result:.6f}"
        )
        subject_results.loc[subject] = [subject_result]
    subject_results.sort_index().to_csv(
        corr_dir / f"trf_{direction}_fs={fs}_lag={lags}_wn={window}{shift_str}.csv",
        index=True,
        index_label="subject",
    )


if __name__ == "__main__":
    main()
