import gzip
from logging import Logger
from pathlib import Path
from typing import Any, List, Dict

import librosa
import numpy as np
from numpy.typing import NDArray

from brain_pipeline.step.step import Step
from brain_pipeline import DefaultKeys, OptionalKey


def fn_librosa(output_keys: List[str], path: Path):
    data, sr = librosa.load(path, sr=None)
    return dict(zip(
        output_keys,
        [
            data.astype(np.float32),
            sr,
        ]
    ))

def fn_npz(output_keys: List[str], path: Path):
    data = np.load(path)
    return dict(zip(
        output_keys,
        [
            data['audio'].astype(np.float32),
            data['fs'].item()
        ]
    ))


def fn_gz(output_keys: List[str], path: Path):
    with gzip.open(path, "rb") as f:
        data: Dict[str, NDArray[np.float32]] = dict(np.load(f))
    return dict(zip(
        output_keys,
        [
            data['audio'].astype(np.float32),
            data['fs'].item()
        ]
    ))


class LoadStimuli(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None):
        super().__init__(
            input_keys,
            [DefaultKeys.I_STI_PATH],
            output_keys,
            [
                DefaultKeys.I_STI_DATA,
                DefaultKeys.I_STI_SR
            ]
        )
        self.assert_keys_num('==', 1, '==', 2)
        self.format_mapping_fn = {
            ".wav": fn_librosa,
            ".mp3": fn_librosa,
            ".npz": fn_npz,
            ".gz": fn_gz,
        }

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Dict[str, NDArray[np.float32]]]:
        path: Path = input_data[self.input_keys[0]]
        ext = path.suffix
        if ext not in self.format_mapping_fn.keys():
            raise KeyError(f"Unsupported file format: {ext}")
        fn = self.format_mapping_fn[ext]
        data = fn(self.output_keys, path)
        logger.info(f"Loaded stimuli from {path.name} with shape {data[self.output_keys[0]].shape}")
        return data


class LoadStimuliTrigger(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None):
        super().__init__(
            input_keys,
            [DefaultKeys.I_STI_TRIGGER_PATH],
            output_keys,
            [
                DefaultKeys.I_STI_TRIGGER_DATA,
                DefaultKeys.I_STI_TRIGGER_SR,
            ]
        )
        self.assert_keys_num('==', 1, '==', 2)
        self.format_mapping_fn = {
            ".wav": fn_librosa,
            ".mp3": fn_librosa,
            ".npz": fn_npz,
            ".gz": fn_gz,
        }

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, NDArray[np.float32]]:
        path: Path = input_data[self.input_keys[0]]
        ext = path.suffix
        if ext not in self.format_mapping_fn.keys():
            raise KeyError(f"Unsupported file format: {ext}")
        fn = self.format_mapping_fn[ext]
        data = fn(self.output_keys, path)
        logger.info(f"Loaded stimuli trigger from {path.name} with shape {data[self.output_keys[0]].shape}")
        return data
