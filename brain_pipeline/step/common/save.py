import os
from logging import Logger
from typing import Any, Callable, List, Dict
from pathlib import Path

import numpy as np
import torch

from brain_pipeline.step.step import Step
from brain_pipeline import Key, DefaultKeys


class FilepathFn:
    def __init__(self, target_root_dir: Path, mode: str = "eeg"):
        """
        Calculate the target file path(s) when saving.

        :param target_root_dir: Root dir of target file path.
        :type target_root_dir: Path
        :param mode: Function type of file path, defaults to "eeg"
        :type mode: str, optional
        :raises KeyError: Unsupported function type.

        Example
        ------
        .. code-block:: python
            Save(
                input_keys=[DefaultKeys.I_STI_PATH, ENVELOPE_256_BOARD_BAND],
                filepath_fn=FilepathFn(DATASET_PROCESSED_DIR, "stimuli"),
            )

        `input_keys` should adjust according to type of `FilepathFn`:
        1. `"eeg"`: `[I_EEG_PATH, I_STI_PATH, EEG_DATA]`
        2. `"stimuli"`: `[I_STI_PATH, STIMULI_DATA]`
        """
        self.target_root_dir = target_root_dir
        self.mode = mode
        self.fn_map = {
            "eeg": self.eeg,
            "stimuli": self.stimuli,
        }
        self.fn: Callable[[List[str], Dict[str, Any]], Dict[str, Path]] | None = (
            self.fn_map.get(self.mode)
        )
        if self.fn is None:
            raise KeyError(f"Unsupported mode: {self.mode}")

    def __call__(self, input_keys: List[str], input_data: Dict[str, Any]):
        if self.fn is None:
            raise ValueError("FilepathFn is not initialized.")
        return self.fn(input_keys, input_data)

    def eeg(self, input_keys: List[str], input_data: Dict[str, Any]):
        """Save EEG

        Example
        ------
        ::

            "E:/Dataset/SparrKULee/sub-001/ses-shortstories01/eeg/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz"
            "E:/Dataset/SparrKULee/stimuli/eeg/audiobook_5_1.npz.gz"
            -> "E:/Dataset/SparrKULee (preprocessed)/eeg-64/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg-64.npy"
        """
        assert len(input_keys) >= 3
        o_eeg_path, o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        subject, session, task, run, ext = o_eeg_path.stem.split("_")
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[2:]:
            target_file_path = (
                self.target_root_dir
                / "eeg"
                / subject
                / f"{session}_{task}_{run}"
                / f"{subject}_{session}_{task}_{run}_{stimuli_name}_{key}.pt"
            )
            ret[key] = target_file_path
        return ret

    def stimuli(self, input_keys: List[str], input_data: Dict[str, Any]):
        """Save stimuli

        Example
        ------
        ::

            "E:/Dataset/SparrKULee/stimuli/eeg/podcast_1.npz.gz"
            -> "E:/Dataset/SparrKULee (preprocessed)/envelope/podcast_1_envelope.npy"
        """
        assert len(input_keys) >= 2
        o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[1:]:
            target_file_path = (
                self.target_root_dir
                / "stimuli"
                / stimuli_name
                / f"{stimuli_name}_{key}.pt"
            )
            ret[key] = target_file_path
        return ret


class Save(Step):
    def __init__(
        self, input_keys: Key, filepath_fn: FilepathFn, ext: str = ".pt", override=True
    ):
        """Save file

        :param input_keys: Key of data to save
        :type input_keys: Key
        :param filepath_fn: A FilepathFn
        :type filepath_fn: FilepathFn
        :param ext: Data format, supports `.pt (torch.Tensor)` and `.npy (np.ndarray)`, defaults to `.pt`
        :type ext: str, optional
        :param overrite: Whether to override the existing file, defaults to True
        :type overrite: bool, optional
        """
        super().__init__(input_keys, [], None, [DefaultKeys.RETURN_CODE])
        self.assert_keys_num(">", 0, "==", 1)
        self.filepath_fn = filepath_fn
        self.ext = ext
        self.override = override

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        target_file_paths = self.filepath_fn(self.input_keys, input_data)
        for key, target_file_path in target_file_paths.items():
            data = input_data[key]
            target_file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            if self.ext:
                target_file_path = target_file_path.with_suffix(self.ext)
            if target_file_path.exists() and not self.override:
                logger.info(f"{target_file_path} exists, will not overrite.")
            if self.ext not in [".pt", ".npy"]:
                logger.warning(f"Unsupported extension: {self.ext}, default to .pt.")
                self.ext = ".pt"
            if self.ext == ".pt":
                fd = os.open(
                    target_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o777
                )
                with os.fdopen(fd, "wb") as f:
                    torch.save(torch.from_numpy(data), f)
            elif self.ext == ".npy":
                np.save(target_file_path, data)
        return {self.output_keys[0]: "success"}
