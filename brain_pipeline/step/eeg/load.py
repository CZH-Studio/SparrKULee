import gzip
import shutil
import tempfile
from logging import Logger
from typing import Any, Optional, List, Dict
from pathlib import Path

import mne
import numpy as np

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import DefaultKeys, KeyTypeNoneOk


class LoadEEG(Step):
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None,
                 unit_multiplier: float = 1.0, selected_channels: Optional[List[int]] = None,
                 trigger_channel: str = "Status"):
        super().__init__(
            input_keys,
            [DefaultKeys.INPUT_EEG_PATH],
            output_keys,
            [
                DefaultKeys.INPUT_EEG_DATA,
                DefaultKeys.INPUT_EEG_TRIGGER,
                DefaultKeys.INPUT_EEG_SR,
            ]
        )
        self.assert_keys('==', 1, '==', 3)
        self.format_mapping_fn = {
            "edf": mne.io.read_raw_edf,
            "bdf": mne.io.read_raw_bdf,
            "gdf": mne.io.read_raw_gdf,
        }
        self.unit_multiplier = unit_multiplier
        self.selected_channels = selected_channels
        self.trigger_channel = trigger_channel

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        gzip_path: Path = input_data[self.input_keys[0]]
        # *.bdf.gz
        ext = gzip_path.stem.split(".")[-1]
        if ext not in self.format_mapping_fn.keys():
            logger.error(f"Unsupported file format: {ext}")
            return {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_bdf_path = Path(tmp_dir) / gzip_path.stem
            with gzip.open(gzip_path, "rb") as f_in, open(tmp_bdf_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            raw = self.format_mapping_fn[ext](tmp_bdf_path, preload=True)
        eeg = raw.get_data()
        if self.trigger_channel:
            trigger = raw.get_data(picks=self.trigger_channel)
        else:
            trigger = eeg[-1].astype(np.int32)
        if self.selected_channels is not None:
            selected_eeg = eeg[self.selected_channels] * self.unit_multiplier
        else:
            selected_eeg = eeg * self.unit_multiplier
        sr = raw.info["sfreq"]
        logger.info(f"Loaded EEG from {gzip_path.name} with shape {selected_eeg.shape}, {sr} Hz.")
        return dict(zip(
            self.output_keys,
            [
                selected_eeg,
                trigger,
                sr,
            ]
        ))
