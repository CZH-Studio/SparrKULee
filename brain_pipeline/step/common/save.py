from logging import Logger
from typing import Any, Callable, List, Dict
from pathlib import Path

import numpy as np

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import KeyTypeNotNone, DefaultKeys


class TargetFilePathFnFactory:
    def __init__(self, target_root_dir: Path, mode: str = 'eeg'):
        self.target_root_dir = target_root_dir
        self.mode = mode
        self.fn_map = {
            "eeg": self.eeg,
            "stimulus": self.stimulus
        }
        self.fn: Callable[[List[str], Dict[str, Any]], Dict[str, Path]] = self.fn_map.get(self.mode)
        if self.fn is None:
            raise KeyError(f"Unsupported mode: {self.mode}")

    def __call__(self, input_keys: List[str], input_data: Dict[str, Any]):
        return self.fn(input_keys, input_data)

    def eeg(self, input_keys: List[str], input_data: Dict[str, Any]):
        """
        "E:/Dataset/SparrKULee/sub-001/ses-shortstories01/eeg/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz"
        "E:/Dataset/SparrKULee/stimuli/eeg/audiobook_5_1.npz.gz"
        -> "E:/Dataset/SparrKULee (preprocessed)/eeg-64/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg-64.npy"
        -> "E:/Dataset/SparrKULee (preprocessed)/eeg-512/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg-512.npy"
        :return:
        """
        o_eeg_path, o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        subject, session, task, run, ext = o_eeg_path.stem.split("_")
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[2:]:
            target_file_path = self.target_root_dir / key / subject / session / f"{subject}_{session}_{task}_{run}_{stimuli_name}_{key}.npy"
            ret[key] = target_file_path
        return ret

    def stimulus(self, input_keys: List[str], input_data: Dict[str, Any]):
        """
        "E:/Dataset/SparrKULee/stimuli/eeg/podcast_1.npz.gz"
        -> "E:/Dataset/SparrKULee (preprocessed)/envelope/podcast_1_envelope.npy"
        -> "E:/Dataset/SparrKULee (preprocessed)/ffr/podcast_1_ffr.npy"
        """
        o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[1:]:
            target_file_path = self.target_root_dir / key / f"{stimuli_name}_{key}.npy"
            ret[key] = target_file_path
        return ret


class Save(Step):
    def __init__(self, input_keys: KeyTypeNotNone, target_file_path_fn: TargetFilePathFnFactory):
        """
        :param input_keys: recommended
        :param target_file_path_fn: 用于生成目标路径的函数
        """
        super().__init__(
            input_keys,
            [],
            None,
            [DefaultKeys.OUTPUT_STATUS]
        )
        self.assert_keys('>', 0, '==', 1)
        self.target_file_path_fn = target_file_path_fn

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        target_file_paths = self.target_file_path_fn(self.input_keys, input_data)
        for key, target_file_path in target_file_paths.items():
            data = input_data[key]
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(target_file_path, data)
        return {self.output_keys[0]: "success"}
