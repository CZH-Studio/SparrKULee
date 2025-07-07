from logging import Logger
from typing import Any, Union, Dict

import numpy as np
from numpy.typing import NDArray
import scipy

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import KeyTypeNoneOk, DefaultKeys


class InterpolateArtifacts(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None,
                 threshold: Union[int, float] = 500):
        super().__init__(
            input_keys,
            [DefaultKeys.FILTERED_DATA],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA]
        )
        self.assert_keys('==', 1, '==', 1)
        self.threshold = threshold

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg = input_data[self.input_keys[0]]
        for channel_index in range(eeg.shape[0]):
            artifact_indices = np.abs(eeg[channel_index, :]) > self.threshold
            concat = np.concatenate(([0], artifact_indices, [0]), axis=0)
            diff = np.diff(concat)
            rising_edges = diff[:-1]
            falling_edges = diff[1:]
            indices = np.arange(eeg.shape[1])
            start_indices = indices[rising_edges == 1]
            stop_indices = indices[falling_edges == -1]
            for start_index, stop_index in zip(start_indices, stop_indices):
                start_sample = eeg[channel_index, start_index]
                stop_sample = eeg[channel_index, stop_index]
                eeg[channel_index, start_index + 1: stop_index] = np.linspace(
                    start_sample, stop_sample, max(stop_index - start_index - 1, 0)
                )
        return {self.output_keys[0]: eeg}


class Align(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None, ):
        super().__init__(
            input_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
                DefaultKeys.INPUT_EEG_TRIGGER,
                DefaultKeys.INPUT_EEG_SR,
                DefaultKeys.INPUT_STIMULI_TRIGGER_DATA,
                DefaultKeys.INPUT_STIMULI_TRIGGER_SR
            ],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA]
        )
        self.assert_keys('==', 5, '==', 1)
        self.logger = None

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg, eeg_trigger, eeg_sr, stimuli_trigger, stimuli_sr = [input_data[key] for key in self.input_keys]
        self.logger = logger
        # biosemi
        triggers = eeg_trigger.flatten().astype(np.int32) & (2 ** 16 - 1)
        values, counts = np.unique(triggers, return_counts=True)
        valid_mask = (0 < values) & (values < 256)
        val_indices = np.argsort(counts[valid_mask])
        most_common = values[valid_mask][val_indices[-1]]
        eeg_trigger = np.int32(triggers != most_common)
        # get trigger indices
        eeg_trigger_indices = self._get_trigger_indices(eeg_trigger)
        # get stimuli indices
        stimuli_trigger_indices = self._get_trigger_indices(stimuli_trigger)
        new_eeg = self._drift_correction(
            eeg, eeg_trigger_indices, eeg_sr,
            stimuli_trigger_indices, stimuli_sr
        )
        return dict(zip(
            self.output_keys,
            [new_eeg]
        ))

    @staticmethod
    def _get_trigger_indices(triggers):
        all_indices = np.where(triggers > 0.5)[0]
        diff_trigger_indices = all_indices[1:] - all_indices[:-1]
        # Keep only the gaps between triggers, not the duration of triggers
        indices_to_keep = diff_trigger_indices > 1
        # Assumption that the EEG doesn't start with a trigger
        # in the first sample (shouldn't be the case)
        return all_indices[np.concatenate(([True], indices_to_keep))]

    def _drift_correction(self, eeg, eeg_trigger_indices, eeg_sr, stimulus_trigger_indices, stimuli_sr):
        """Correct the drift between the brain response data and the stimulus.

            When the brain response data and the stimulus data are not recorded on
            the same system (i.e. using the same clock), clock drift may cause the
            brain response data to be misaligned with the stimulus. This function
            tries to correct for this by interpolating the brain response data to
            the same length as the stimulus.

            Parameters
            ----------
            eeg: np.ndarray
                The brain response data. The data is expected to be a 2D array of
                shape (n_channels, n_samples).
            eeg_trigger_indices: np.ndarray
                The indices of the brain response data where the triggers are located.
                The data is expected to be a 1D array of integers.
            eeg_sr: int
                The sampling frequency of the brain response data.
            stimulus_trigger_indices: np.ndarray
                The indices of the stimulus data where the triggers are located.
                The data is expected to be a 1D array of integers.
            stimuli_sr: int
                The sampling frequency of the stimulus data.

            Returns
            -------
            np.ndarray
                The brain response data with the same length as the stimulus data.
            """
        # We can fix one missing stimulus trigger by adding one to the brain trigger
        if len(eeg_trigger_indices) + 1 < len(stimulus_trigger_indices):
            raise ValueError(
                    f"Number of triggers does not match "
                    f"(in eeg {len(eeg_trigger_indices)} were found, "
                    f"in stimulus {len(stimulus_trigger_indices)})"
                )
        elif len(eeg_trigger_indices) < len(stimulus_trigger_indices):
            # Check if the first trigger is missing
            last_brain_trigger_duration = (
                                                  eeg_trigger_indices[-1] - eeg_trigger_indices[-2]
                                          ) / eeg_sr
            last_stim_trigger_duration = (
                                                 stimulus_trigger_indices[-1] - stimulus_trigger_indices[-2]
                                         ) / stimuli_sr
            if (
                    (last_brain_trigger_duration * 0.99)
                    < last_stim_trigger_duration  # noqa: W503
                    < (last_brain_trigger_duration * 1.01)  # noqa: W503
            ):
                # The last trigger is missing
                eeg_trigger_indices = np.concatenate(
                    (
                        eeg_trigger_indices,
                        [
                            eeg_trigger_indices[-1]
                            + np.round(  # noqa: W503
                                last_stim_trigger_duration * eeg_sr
                            ).astype(int)
                        ],
                    ),
                    axis=0,
                )
            else:
                # The first trigger is missing
                eeg_trigger_indices = np.concatenate(
                    (
                        [eeg_trigger_indices[0] - eeg_sr],
                        eeg_trigger_indices,
                    ),
                    axis=0,
                )
        elif (
                len(eeg_trigger_indices) == len(stimulus_trigger_indices) + 1
                and eeg_trigger_indices[0] == 0  # noqa: W503
        ):
            # Check if there is an erroneous trigger at the beginning
            eeg_trigger_indices = eeg_trigger_indices[1:]

        stimulus_diff = stimulus_trigger_indices[-1] - stimulus_trigger_indices[0]
        expected_length = int(np.ceil(stimulus_diff / stimuli_sr * eeg_sr))
        real_length = eeg_trigger_indices[-1] - eeg_trigger_indices[0]
        tmp_eeg = eeg[:, eeg_trigger_indices[0]: eeg_trigger_indices[-1]]
        idx_real = np.linspace(0, 1, real_length)
        idx_expected = np.linspace(0, 1, expected_length)
        interpolate_fn = scipy.interpolate.interp1d(idx_real, tmp_eeg, "linear", axis=1)
        new_eeg: NDArray = interpolate_fn(idx_expected)

        new_start = eeg_trigger_indices[0]
        begin_eeg = eeg[:, :new_start]
        end_eeg = eeg[:, eeg_trigger_indices[-1] + 1:]

        new_end = int(eeg_trigger_indices[-1] + 2 * eeg_sr)
        # Make length multiple of samplerate
        new_end = int(np.ceil((new_end - new_start) / eeg_sr) * eeg_sr + new_start - 1)

        total_eeg = begin_eeg[:, new_start:]
        new_eeg_start = max(int(new_start - begin_eeg.shape[1]), 0)
        new_eeg_end = min(new_end - begin_eeg.shape[1], new_eeg.shape[1])
        total_eeg = np.concatenate(
            (total_eeg, new_eeg[:, new_eeg_start:new_eeg_end]), axis=1
        )
        end_eeg_start = max(int(new_start - begin_eeg.shape[1] - new_eeg.shape[1]), 0)
        end_eeg_end = min(new_end - begin_eeg.shape[1] - new_eeg.shape[1], end_eeg.shape[1])
        total_eeg = np.concatenate(
            (total_eeg, end_eeg[:, end_eeg_start:end_eeg_end]), axis=1
        )
        if total_eeg.shape[1] % eeg_sr != 0:
            nb_seconds = np.floor(eeg.shape[1] / eeg_sr)
            total_eeg = total_eeg[:, : int(nb_seconds * eeg_sr)]
        return total_eeg


class RemoveArtifacts(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None,
                 reference_channels=None, delay: int = 3):
        super().__init__(
            input_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
                DefaultKeys.INPUT_EEG_SR,
            ],
            output_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
            ]
        )
        self.assert_keys('==', 2, '==', 1)

        if reference_channels is None:
            self.reference_channels = [0, 1, 2, 32, 33, 34, 35, 36]
        else:
            self.reference_channels = reference_channels
        self.delay = delay
        self.axis = 1

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg, sr = [input_data[key] for key in self.input_keys]
        eeg = eeg.T
        mask = self._get_artifact_segments(eeg, sr)
        mwf_weights = self._compute_mwf(eeg.T, mask)
        filtered_eeg, artifacts = self._apply_mwf(eeg.T, mwf_weights)
        eeg_new = np.real(filtered_eeg)
        return dict(zip(
            self.output_keys,
            [eeg_new]
        ))

    def _get_artifact_segments(self, eeg, fs):
        ref = np.sum(eeg[:, self.reference_channels] ** 2, axis=self.axis)
        threshold = 5 * np.mean(ref)
        mask = ref > threshold
        indices = np.where(mask)[0]
        window_len = int(np.round(fs / 2))
        n_frames = eeg.shape[0]
        for i in range(len(indices)):
            if indices[i] < window_len:
                mask[: indices[i] + window_len + 1] = True
            elif n_frames - indices[i] < window_len:
                mask[indices[i] - window_len :] = True
            else:
                mask[indices[i] - window_len : indices[i] + window_len + 1] = True
        return mask

    def _stack_delayed(self, eeg, delay):
        nb_channels = eeg.shape[0]
        nb_shifted_channels = (2 * delay + 1) * nb_channels
        data_s = np.zeros((nb_shifted_channels, eeg.shape[1]))
        for tau in range(-delay, delay + 1):
            start_ = (tau + delay) * nb_channels
            end_ = (tau + delay + 1) * nb_channels
            shifted = np.roll(eeg, tau, axis=self.axis)
            if tau > 0:
                shifted[:, :tau] = 0
            elif tau < 0:
                shifted[:, tau:] = 0
            data_s[start_:end_, :] = shifted
        return data_s, nb_shifted_channels

    @staticmethod
    def _check_symmetric(data, rtol=1e-05, atol=1e-08):
        return np.allclose(data, data.T, rtol=rtol, atol=atol)

    def _fix_symmetric(self, mat):
        if self._check_symmetric(mat):
            return mat
        else:
            return (mat.T + mat) / 2

    @staticmethod
    def _sort_evd(eig_values, eig_vectors):
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        return eig_values, eig_vectors

    def _compute_mwf(self, eeg, mask):
        data, nb_shifted_channels = self._stack_delayed(eeg, self.delay)
        ryy = self._fix_symmetric(np.cov(data[:, mask]))
        rnn = self._fix_symmetric(np.cov(data[:, ~mask]))
        eig_values, eig_vectors = scipy.linalg.eig(ryy, rnn)
        eig_values, eig_vectors = self._sort_evd(eig_values, eig_vectors)
        temp_eig_values = np.diag(eig_vectors.T @ ryy @ eig_vectors)
        denormalization_factors = np.repeat(
            np.sqrt(temp_eig_values / eig_values)[np.newaxis],
            eig_vectors.shape[0],
            axis=0,
        )
        denorm_eig_vectors = eig_vectors / denormalization_factors
        eig_values_y = denorm_eig_vectors.T @ ryy @ denorm_eig_vectors
        eig_values_n = denorm_eig_vectors.T @ rnn @ denorm_eig_vectors
        delta = eig_values_y - eig_values_n
        rank_w = nb_shifted_channels - np.sum(np.diagonal(delta) < 0)
        eig_values_to_truncate = range(rank_w, delta.shape[1])
        indices = (eig_values_to_truncate, eig_values_to_truncate)
        delta[indices] = 0
        eig_values_mat = eig_values * np.eye(delta.shape[0])
        left = -np.linalg.solve(eig_values_mat.T, denorm_eig_vectors.T).T
        right = -np.linalg.solve(denorm_eig_vectors.T, delta.T).T
        return left @ right

    def _apply_mwf(self, eeg, mwf_weights):
        channels, time = eeg.shape
        nb_weights = mwf_weights.shape[0]
        tau = (nb_weights - channels) // (2 * channels)
        channel_means = eeg.mean(axis=self.axis, keepdims=True)
        eeg = eeg - channel_means
        eeg_shifted, _ = self._stack_delayed(eeg, tau)
        orig_chans = range(tau * channels, (tau + 1) * channels)
        artifacts = mwf_weights[:, orig_chans].T @ eeg_shifted
        filtered_eeg = eeg - artifacts
        return filtered_eeg + channel_means, artifacts


class CommonAverageReference(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None,
                 axis: int = 0):
        super().__init__(
            input_keys,
            [DefaultKeys.TMP_EEG_DATA],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA],
        )
        self.assert_keys('==', 1, '==', 1)
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg = input_data[self.input_keys[0]]
        eeg = eeg - np.mean(eeg, axis=self.axis, keepdims=True)
        return dict(zip(
            self.output_keys,
            [eeg]
        ))