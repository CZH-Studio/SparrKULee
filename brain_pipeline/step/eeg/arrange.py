from logging import Logger
from typing import Any, Union, Dict

import numpy as np
from numpy.typing import NDArray
import scipy

from brain_pipeline.step.step import Step
from brain_pipeline import OptionalKey, DefaultKeys


class InterpolateArtifacts(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None,
                 threshold: Union[int, float] = 500):
        super().__init__(
            input_keys,
            [DefaultKeys.FILTERED_DATA],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA]
        )
        self.assert_keys_num('==', 1, '==', 1)
        self.threshold = threshold

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg = input_data[self.input_keys[0]]
        num_interpolated = 0
        for channel_index in range(eeg.shape[0]):
            # 提取出电压大于阈值的伪影成分索引(布尔值数组)
            artifact_indices = np.abs(eeg[channel_index]) > self.threshold
            # 计算出现伪影成分的起始和终止边界(索引都在伪影成分内部)
            diff = np.diff(np.concatenate(([0], artifact_indices, [0]), axis=0))
            border_indices = np.column_stack((np.where(diff == 1)[0], np.where(diff == -1)[0] - 1))
            num_interpolated += border_indices.shape[0]
            # 执行插值伪影算法，将伪影成分内的采样点线性插值到边界点
            for start_index, stop_index in border_indices:
                start_sample = eeg[channel_index, start_index]
                stop_sample = eeg[channel_index, stop_index]
                eeg[channel_index, start_index + 1: stop_index] = np.linspace(
                    start_sample, stop_sample, max(stop_index - start_index - 1, 0)
                )
        logger.info(f"Interpolated {num_interpolated} artifact samples in all channels.")
        return {self.output_keys[0]: eeg}


class Align(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None, ):
        """
        Align the EEG data to the stimulus data.
        :param input_keys: 期望：EEG数据、EEG触发、EEG采样率、刺激触发、刺激采样率
        :param output_keys: 对齐后的EEG数据
        """
        super().__init__(
            input_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
                DefaultKeys.I_EEG_TRIGGER,
                DefaultKeys.I_EEG_SR,
                DefaultKeys.I_STI_TRIGGER_DATA,
                DefaultKeys.I_STI_TRIGGER_SR
            ],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA]
        )
        self.assert_keys_num('==', 5, '==', 1)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg, eeg_trigger, eeg_sr, stimuli_trigger, stimuli_sr = [input_data[key] for key in self.input_keys]
        # biosemi触发信号格式
        triggers = eeg_trigger.flatten().astype(np.int32) & 0xFFFF  # EEG触发信号中，取最后16位有效位
        clipped = np.where((triggers >= 1) & (triggers <= 255), triggers, 0)  # 截断到1-255范围
        counts = np.bincount(clipped, minlength=256)  # 统计各触发信号出现次数
        most_common_trigger = np.argmax(counts[1:256]) + 1
        eeg_trigger = (triggers != most_common_trigger).astype(np.int32)  # 标记非最常触发信号的位置
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
    def _get_trigger_indices(triggers: NDArray):
        # 获取触发信号为1的索引
        all_indices = np.where(triggers > 0.5)[0]  # 这个地方写大于0.5而不是直接等于1是因为存在浮点数误差
        # 获取触发信号的起始边界
        return all_indices[np.concatenate(([True], np.diff(all_indices) > 1))]

    @staticmethod
    def _drift_correction(
            eeg: NDArray,
            eeg_trigger_indices: NDArray,
            eeg_sr: int,
            stimulus_trigger_indices: NDArray,
            stimuli_sr: int
    ):
        delta_num_triggers = len(eeg_trigger_indices) - len(stimulus_trigger_indices)
        # 如果EEG触发器的数量明显少于刺激触发器的数量，则抛出异常
        if delta_num_triggers < -1:
            raise ValueError(
                f"Number of triggers does not match "
                f"(in eeg {len(eeg_trigger_indices)} were found, "
                f"in stimulus were {len(stimulus_trigger_indices)})"
            )
        # 如果EEG触发器的数量+1等于刺激触发器的数量，则可以补全
        elif delta_num_triggers == -1:
            # Check if the first trigger is missing
            last_eeg_trigger_duration = (eeg_trigger_indices[-1] - eeg_trigger_indices[-2]) / eeg_sr
            last_stim_trigger_duration = (stimulus_trigger_indices[-1] - stimulus_trigger_indices[-2]) / stimuli_sr
            if 0.99 < last_stim_trigger_duration / last_eeg_trigger_duration < 1.01:
                # The last trigger is missing
                eeg_trigger_indices = np.concatenate(
                    (
                        eeg_trigger_indices,
                        [eeg_trigger_indices[-1] + np.round(last_stim_trigger_duration * eeg_sr).astype(int)],
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
        elif delta_num_triggers == 1 and eeg_trigger_indices[0] == 0:
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
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None,
                 reference_channels=None, delay: int = 3):
        super().__init__(
            input_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
                DefaultKeys.I_EEG_SR,
            ],
            output_keys,
            [
                DefaultKeys.TMP_EEG_DATA,
            ]
        )
        self.assert_keys_num('==', 2, '==', 1)

        if reference_channels is None:
            self.reference_channels = [0, 1, 2, 32, 33, 34, 35, 36]
        else:
            self.reference_channels = reference_channels
        self.delay = delay
        self.axis = 1

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg, sr = [input_data[key] for key in self.input_keys]
        eeg: NDArray
        sr: int
        eeg = eeg.T
        mask = self._get_artifact_segments(eeg, sr)
        mwf_weights = self._compute_mwf(eeg.T, mask)
        filtered_eeg, artifacts = self._apply_mwf(eeg.T, mwf_weights)
        eeg_new = np.real(filtered_eeg)
        return dict(zip(
            self.output_keys,
            [eeg_new]
        ))

    def _get_artifact_segments(self, eeg: NDArray, fs: int):
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
                mask[indices[i] - window_len:] = True
            else:
                mask[indices[i] - window_len: indices[i] + window_len + 1] = True
        return mask

    def _stack_delayed(self, eeg: NDArray, delay: int):
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
    def _check_symmetric(data: NDArray, rtol=1e-05, atol=1e-08):
        return np.allclose(data, data.T, rtol=rtol, atol=atol)

    def _fix_symmetric(self, mat: NDArray):
        if self._check_symmetric(mat):
            return mat
        else:
            return (mat.T + mat) / 2

    @staticmethod
    def _sort_evd(eig_values: NDArray, eig_vectors: NDArray):
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]
        return eig_values, eig_vectors

    def _compute_mwf(self, eeg: NDArray, mask: NDArray):
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

    def _apply_mwf(self, eeg: NDArray, mwf_weights: NDArray):
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
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None,
                 axis: int = 0):
        super().__init__(
            input_keys,
            [DefaultKeys.TMP_EEG_DATA],
            output_keys,
            [DefaultKeys.TMP_EEG_DATA],
        )
        self.assert_keys_num('==', 1, '==', 1)
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        eeg = input_data[self.input_keys[0]]
        eeg = eeg - np.mean(eeg, axis=self.axis, keepdims=True)
        return dict(zip(
            self.output_keys,
            [eeg]
        ))
