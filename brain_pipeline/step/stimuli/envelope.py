from logging import Logger
from typing import Any, Dict

import numpy as np
from brian2 import Hz
from brian2hears import Sound, erbspace, Gammatone, Filterbank

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import DefaultKeys, KeyTypeNoneOk


class EnvelopeCalculator(Filterbank):
    def __init__(self, source, power_factor):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filter bank output to convert to envelope
        power_factor : float
            The power factor for each sample.
        """
        super().__init__(source)
        self.power_factor = power_factor
        self.nchannels = 1

    def buffer_apply(self, input_):  # noqa: D102
        return np.reshape(
            np.sum(np.power(np.abs(input_), self.power_factor), axis=1, keepdims=True),
            (np.shape(input_)[0], self.nchannels),
        )


class GammatoneEnvelope(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None,
                 power_factor: float = 0.6, min_freq: int = 50,
                 max_freq: int = 5000, bands: int = 28):
        super().__init__(
            input_keys,
            [
                DefaultKeys.INPUT_STIMULI_DATA,
                DefaultKeys.INPUT_STIMULI_SR,
            ],
            output_keys,
            [DefaultKeys.ENVELOPE_DATA]
        )
        self.assert_keys('==', 2, '==', 1)
        self.power_factor = power_factor
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.bands = bands

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        stimuli_data, stimuli_sr = [input_data[key] for key in self.input_keys]
        if len(stimuli_data.shape) == 1:
            stimuli_data = np.expand_dims(stimuli_data, axis=1)
        sound = Sound(stimuli_data, stimuli_sr * Hz)
        center_frequencies = erbspace(
            self.min_freq * Hz, self.max_freq * Hz, self.bands
        )
        filter_bank = Gammatone(sound, center_frequencies)
        envelope = EnvelopeCalculator(filter_bank, self.power_factor)
        output = envelope.process()
        return dict(zip(
            self.output_keys,
            [output]
        ))
