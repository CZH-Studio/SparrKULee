from logging import Logger
from typing import Any, Dict

from scipy import signal

from brain_pipeline.step.step import Step
from brain_pipeline import Key, OptionalKey, DefaultKeys


class ResamplePoly(Step):
    def __init__(
        self, input_keys: Key, output_keys: OptionalKey, target_sr: int, axis: int = 0
    ):
        """Resample data using polyphase filtering.

        :param input_keys: `[data, sr]`
        :type input_keys: Key
        :param output_keys: resampled data and sr, default to `[RESAMPLED_DATA, RESAMPLED_SR]`
        :type output_keys: OptionalKey
        :param target_sr: resample sr
        :type target_sr: int
        :param axis: resample axis, defaults to 0
        :type axis: int, optional
        """
        super().__init__(
            input_keys,
            [],
            output_keys,
            [DefaultKeys.RESAMPLED_DATA, DefaultKeys.RESAMPLED_SR],
        )
        self.assert_keys_num("==", 2, "==", 2)
        self.target_sr = target_sr
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger):
        data, original_sr = [input_data[k] for k in self.input_keys]
        data = signal.resample_poly(data, self.target_sr, original_sr, axis=self.axis)
        return dict(zip(self.output_keys, [data, self.target_sr]))


class SosFilter(Step):
    def __init__(
        self,
        input_keys: Key,
        output_keys: OptionalKey,
        n: int,
        window,
        btype: str,
        axis: int = -1,
    ):
        """Apply zero-phase filtering to the data.

        :param input_keys: `[data, sr]`
        :type input_keys: Key
        :param output_keys: `[FILTERED_DATA]`
        :type output_keys: OptionalKey
        :param n: Filter order (higher means steeper, but also less stable)
        :type n: int
        :param window: For lowpass and highpass filters, it is a scalar; for bandpass and bandstop filters, it is a length-2 sequence.
        :type window: scalar / length-2 sequence
        :param btype: Filter type
        :type btype: str
        :param axis: Filter axis, defaults to -1
        :type axis: int, optional
        """
        super().__init__(
            input_keys,
            [],
            output_keys,
            [DefaultKeys.FILTERED_DATA],
        )
        self.assert_keys_num("==", 2, "==", 1)
        self.n = n
        self.window = window
        self.btype = btype
        self.axis = axis

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        data, sr = [input_data[k] for k in self.input_keys]
        # 设计一个滤波器
        sos = signal.butter(self.n, self.window, self.btype, fs=sr, output="sos")
        # 应用滤波器
        data = signal.sosfiltfilt(sos, data, axis=self.axis)
        return dict(zip(self.output_keys, [data]))
