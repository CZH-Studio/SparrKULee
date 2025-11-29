from logging import Logger
from typing import Any, Dict

from scipy import signal

from brain_pipeline.step.step import Step
from brain_pipeline import Key, OptionalKey, DefaultKeys


class ResamplePoly(Step):
    def __init__(self, input_keys: Key, output_keys: OptionalKey,
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
        self.assert_keys_num('==', 2, '==', 1)
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
    def __init__(self, input_keys: OptionalKey, output_keys: OptionalKey,
                 n: int, window, btype: str, axis: int = -1):
        """
        对数据应用零相位滤波
        :param input_keys: [data, sr]
        :param output_keys: [filtered_data]
        :param n: 滤波器阶数（越高越陡，但也更不稳定）
        :param window: 窗口
        :param btype: 滤波类型
        :param axis: default is -1 (last dimension)
        """
        super().__init__(
            input_keys,
            [
                DefaultKeys.I_EEG_DATA,
                DefaultKeys.I_EEG_SR
            ],
            output_keys,
            [DefaultKeys.FILTERED_DATA]
        )
        self.assert_keys_num('==', 2, '==', 1)
        self.n = n
        self.window = window
        self.btype = btype
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        data, sr = [input_data[k] for k in self.input_keys]
        # 设计一个滤波器
        sos = signal.butter(self.n, self.window, self.btype, fs=sr, output='sos')
        # 应用滤波器
        data = signal.sosfiltfilt(sos, data, axis=self.axis)
        return dict(zip(
            self.output_keys,
            [data]
        ))
