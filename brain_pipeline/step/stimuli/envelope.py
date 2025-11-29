from logging import Logger
from typing import Any, Dict

import numpy as np
from brian2 import Hz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

from brain_pipeline.step.step import Step
from brain_pipeline import DefaultKeys, OptionalKey


class EnvelopeCalculator(Filterbank):
    def __init__(self, source, power_factor):
        super().__init__(source)
        self.power_factor = power_factor
        self.nchannels = 1

    def buffer_apply(self, input):
        return np.reshape(
            np.sum(np.power(np.abs(input), self.power_factor), axis=1, keepdims=True),
            (np.shape(input)[0], self.nchannels),
        )


class GammatoneEnvelope(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None,
                 power_factor: float = 0.6, min_freq: int = 50,
                 max_freq: int = 5000, bands: int = 28):
        """
        语音包络（envelope）是指声音信号的能量分布，即声音信号的能量随时间的变化情况。
        语音包络的计算没有严格的定义，但最佳实践是使用Gammatone滤波器组，它能够提取声音信号的不同频率成分，符合人耳的听觉处理。
        语音包络的计算可以分为以下步骤：
        1. 读取语音信号
        2. 初始化Gammatone滤波器组，默认包含28个滤波器，中心频率在50Hz到5000Hz之间
        3. 将语音信号经过滤波器组进行滤波操作，得到滤波后的语音信号组
        4. 对于滤波后的语音信号组（多通道），计算其绝对值，乘以0.6次方，再将各个通道求和，得到包络信号（单通道）
        详见 https://ieeexplore.ieee.org/document/7478117
        :param input_keys: 输入
        :param output_keys: 输出
        :param power_factor: 幅值
        :param min_freq: 最低频率
        :param max_freq: 最高频率
        :param bands: 组中滤波器的数量
        """
        super().__init__(
            input_keys,
            [
                DefaultKeys.I_STI_DATA,
                DefaultKeys.I_STI_SR,
            ],
            output_keys,
            [DefaultKeys.ENVELOPE_DATA]
        )
        self.assert_keys_num('==', 2, '==', 1)
        self.power_factor = power_factor
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bands = bands

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        stimuli_data, stimuli_sr = [input_data[key] for key in self.input_keys]
        if len(stimuli_data.shape) == 1:
            stimuli_data = np.expand_dims(stimuli_data, axis=1)
        sound = Sound(stimuli_data, stimuli_sr * Hz)
        # erbspace 函数生成一个在对数尺度上均匀分布的中心频率数组，这个数组被用来初始化Gammatone滤波器组。
        # 每个滤波器在该频带中心频率处响应最大，从而能够有效地提取该频带的特征。
        center_frequencies = erbspace(self.min_freq * Hz, self.max_freq * Hz, self.bands)
        # Gammatone滤波器组的设计灵感来源于人类耳蜗的机械和生物特性。它能够更好地模拟人耳对不同频率的响应，尤其是在高频区域。
        # Gammatone滤波器组的中心频率在对数尺度上分布，这意味着频率间隔在听觉感知上更加均匀。
        # 每个Gammatone滤波器是一个带通滤波器，具有一个中心频率和一个带宽。滤波器在中心频率处响应最大，在中心频率两侧的频率响应逐渐下降。
        filter_bank = Gammatone(sound, center_frequencies)  # 滤波器组
        envelope = EnvelopeCalculator(filter_bank, self.power_factor)  # 计算语音包络
        output = envelope.process()
        return dict(zip(
            self.output_keys,
            [output]
        ))
