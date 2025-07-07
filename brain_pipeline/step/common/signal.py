from logging import Logger
from typing import Any, Dict

from scipy import signal

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import KeyTypeNotNone, KeyTypeNoneOk, DefaultKeys


class ResamplePoly(Step):
    def __init__(self, input_keys: KeyTypeNotNone, output_keys: KeyTypeNoneOk,
                 target_sr: int, axis: int = 0):
        """
        Resample data using polyphase filtering.
        :param input_keys: [data, input_sr]
        :param output_keys: [resampled_data]
        :param target_sr: target sampling rate
        """
        super().__init__(
            input_keys,
            [],
            output_keys,
            [DefaultKeys.RESAMPLED_DATA]
        )
        self.assert_keys('==', 2, '==', 1)
        self.target_sr = target_sr
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger):
        data, original_sr = [input_data[k] for k in self.input_keys]
        data = signal.resample_poly(data, self.target_sr, original_sr, axis=self.axis)
        return dict(zip(
            self.output_keys,
            [data]
        ))


class SosFilter(Step):
    def __init__(self, input_keys: KeyTypeNoneOk, output_keys: KeyTypeNoneOk,
                 n: int, window, btype: str, axis: int = -1):
        """
        Filter data using second-order sections (SOS) filter.
        :param input_keys: [eeg_data, eeg_sr]
        :param output_keys: [filtered_data]
        :param n: number of filter coefficients
        :param window: window function
        :param btype: bandpass / highpass / lowpass
        :param axis: default is -1 (last dimension)
        """
        super().__init__(
            input_keys,
            [
                DefaultKeys.INPUT_EEG_DATA,
                DefaultKeys.INPUT_EEG_SR
            ],
            output_keys,
            [DefaultKeys.FILTERED_DATA]
        )
        self.assert_keys('==', 2, '==', 1)
        self.n = n
        self.window = window
        self.btype = btype
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        data, sr = [input_data[k] for k in self.input_keys]
        sos = signal.butter(self.n, self.window, self.btype, fs=sr, output='sos')
        data = signal.sosfiltfilt(sos, data, axis=self.axis)
        return dict(zip(
            self.output_keys,
            [data]
        ))
