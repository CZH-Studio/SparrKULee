import gzip
import shutil
import tempfile
from logging import Logger
from typing import Any, Optional, List, Dict
from pathlib import Path

import mne
import numpy as np

from brain_pipeline.step.step import Step
from brain_pipeline import DefaultKeys, OptionalKey


class LoadEEG(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None,
                 unit_multiplier: float = 1.0, selected_channels: Optional[List[int]] = None,
                 trigger_channel: str = "Status"):
        """
        加载EEG数据（包括数据本身、采样率和触发信号），可以选择单位变换，以及选择通道
        :param input_keys:
        :param output_keys:
        :param unit_multiplier:
        :param selected_channels:
        :param trigger_channel:
        """
        super().__init__(
            input_keys,
            [DefaultKeys.I_EEG_PATH],
            output_keys,
            [
                DefaultKeys.I_EEG_DATA,
                DefaultKeys.I_EEG_TRIGGER,
                DefaultKeys.I_EEG_SR,
            ]
        )
        self.assert_keys_num('==', 1, '==', 3)
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
            raw = self.format_mapping_fn[ext](tmp_bdf_path, preload=True, verbose=False)
        eeg = raw.get_data()
        # 有关trigger数据：默认从EEG的Status通道中获取，如果没有则使用最后一个通道作为触发信号
        # 触发信号包括252、254、65788三个数值
        if self.trigger_channel:
            trigger = raw.get_data(picks=self.trigger_channel)  # (1, T)
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
