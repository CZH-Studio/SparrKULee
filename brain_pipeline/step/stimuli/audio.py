from logging import Logger
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import torch
import librosa
from brian2 import Hz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
import scipy
from transformers import Wav2Vec2ForCTC

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
    def __init__(
        self,
        input_keys: OptionalKey = None,
        output_keys: OptionalKey = None,
        power_factor: float = 0.6,
        min_freq: int = 50,
        max_freq: int = 5000,
        bands: int = 28,
    ):
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
            [DefaultKeys.ENVELOPE_DATA],
        )
        self.assert_keys_num("==", 2, "==", 1)
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
        center_frequencies = erbspace(
            self.min_freq * Hz, self.max_freq * Hz, self.bands
        )
        # Gammatone滤波器组的设计灵感来源于人类耳蜗的机械和生物特性。它能够更好地模拟人耳对不同频率的响应，尤其是在高频区域。
        # Gammatone滤波器组的中心频率在对数尺度上分布，这意味着频率间隔在听觉感知上更加均匀。
        # 每个Gammatone滤波器是一个带通滤波器，具有一个中心频率和一个带宽。滤波器在中心频率处响应最大，在中心频率两侧的频率响应逐渐下降。
        filter_bank = Gammatone(sound, center_frequencies)  # 滤波器组
        envelope = EnvelopeCalculator(filter_bank, self.power_factor)  # 计算语音包络
        output = envelope.process()
        return dict(zip(self.output_keys, [output]))


class MelSpectrogram(Step):
    def __init__(
        self,
        input_keys: OptionalKey = None,
        output_keys: OptionalKey = None,
        power_factor: float = 0.6,
        target_sr: int = 64,
        hop_length=None,
        win_length_sec=0.025,
        n_fft=None,
        window_fn=None,
        n_mels=28,
        fmin=-4.2735,
        fmax=5444,
        power=1.0,
        center=False,
        norm=None,
        htk=True,
    ):
        super().__init__(
            input_keys,
            [DefaultKeys.I_STI_DATA, DefaultKeys.I_STI_SR],
            output_keys,
            [DefaultKeys.MEL_DATA, DefaultKeys.MEL_SR],
        )
        self.assert_keys_num("==", 2, "==", 2)
        self.power_factor = power_factor
        self.target_sr = target_sr
        self.hop_length = hop_length
        self.win_length_sec = win_length_sec
        self.n_fft = n_fft
        self.window_fn = window_fn
        if window_fn is None:
            self.window_fn = scipy.signal.windows.hamming
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.center = center
        self.norm = norm
        self.htk = htk

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        stimuli_data, stimuli_sr = [input_data[key] for key in self.input_keys]
        # compute mel kwargs
        mel_kwargs = {
            "fmin": self.fmin,
            "fmax": self.fmax,
            "n_mels": self.n_mels,
            "power": self.power,
            "center": self.center,
            "norm": self.norm,
            "htk": self.htk,
            "hop_length": (
                self.hop_length
                if self.hop_length is not None
                else int((1 / self.target_sr) * stimuli_sr)
            ),
            "win_length": (
                self.win_length_sec
                if self.win_length_sec is None
                else int(self.win_length_sec * stimuli_sr)
            ),
        }
        mel_kwargs["n_fft"] = (
            self.n_fft
            if self.n_fft is not None
            else int(2 ** np.ceil(np.log2(mel_kwargs["win_length"])))
        )
        assert self.window_fn is not None
        mel_kwargs["window"] = self.window_fn(mel_kwargs["win_length"])

        stimuli_data = stimuli_data - np.mean(stimuli_data)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=stimuli_data, sr=stimuli_sr, **mel_kwargs
        ).T
        mel_spectrogram = np.power(mel_spectrogram, self.power_factor)
        return dict(zip(self.output_keys, [mel_spectrogram, mel_kwargs["hop_length"]]))


class Wav2Vec(Step):
    def __init__(
        self,
        input_keys: OptionalKey = None,
        output_keys: OptionalKey = None,
        model_name: str = "",
        lang: str = "en",
        extract_layers: Optional[list[int]] = None,
        overlap: int = 2,
        segment_length: int = 8,
        target_sr: int = 64,
    ):
        super().__init__(
            input_keys,
            [DefaultKeys.I_STI_DATA, DefaultKeys.I_STI_SR],
            output_keys,
            [DefaultKeys.TMP_STIMULI_DATA],
        )
        self.assert_keys_num("==", 2, ">=", 1)
        self.model_name = model_name
        self.lang = lang
        self.extract_layers = extract_layers if extract_layers is not None else [19]
        assert len(self.extract_layers) == len(
            self.output_keys
        ), "Number of output keys should match number of layers to extract"
        self.overlap = overlap
        self.segment_length = segment_length
        self.target_sr = target_sr
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        return self._model

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        audio_data, audio_sr = [input_data[key] for key in self.input_keys]
        audio_data: NDArray
        audio_sr: int

        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        segment_length = self.segment_length * audio_sr
        audio_length = np.size(audio_data)

        audio_data = np.concatenate(
            [
                np.zeros((1, int(self.overlap / 2) * audio_sr), dtype=np.float32),
                audio_data,
            ],
            axis=1,
        )
        eof = False
        outputs = {}
        for layer in self.extract_layers:
            outputs[layer] = []
        for i in range(int(audio_length / segment_length) + 1):
            start = i * segment_length
            end = start + segment_length + self.overlap * audio_sr

            if end < np.size(audio_data):
                speech_segment = audio_data[:, start:end]
            else:
                speech_segment = audio_data[:, start:]
                eof = True
            input = torch.tensor(speech_segment)
            with torch.no_grad():
                logits = self.model.base_model(
                    input,
                    attention_mask=torch.ones_like(input),
                    output_hidden_states=True,
                )
            for layer in self.extract_layers:
                out = logits["hidden_states"][layer]
                out = out.numpy()
                out = np.squeeze(out)
                if eof:
                    out = out[int(self.overlap / 2) * 50 :]
                else:
                    out = out[
                        int(self.overlap / 2) * 50 : -int(self.overlap / 2) * 50 + 1, :
                    ]
                outputs[layer].append(out)
        for k, v in outputs.items():
            stacked = np.vstack(v)
            num_samples = round(np.size(stacked, axis=0) * float(self.target_sr) / 50)
            stacked = scipy.signal.resample(stacked, num_samples)
            outputs[k] = stacked
        output_list = list(outputs.values())
        return dict(zip(self.output_keys, output_list))
