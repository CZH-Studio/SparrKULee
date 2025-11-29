import os
from logging import Logger
from typing import Any, Callable, List, Dict
from pathlib import Path

import numpy as np
import torch

from brain_pipeline.step.step import Step
from brain_pipeline import Key, DefaultKeys


class FilepathFn:
    def __init__(self, target_root_dir: Path, mode: str = 'eeg'):
        """
        在保存时，根据输入的路径生成目标路径
        在context中，源文件的路径key在前面放置，目标文件的路径key在后面放置
        比如当模式采用eeg时，输入的路径key应当分别为：[INPUT_EEG_PATH, INPUT_STIMULI_PATH, EEG_DATA]
        :param target_root_dir: 输出根目录
        :param mode: 选择目标函数
        """
        self.target_root_dir = target_root_dir
        self.mode = mode
        self.fn_map = {
            "eeg": self.eeg,
            "stimulus": self.stimulus,
        }
        self.fn: Callable[[List[str], Dict[str, Any]], Dict[str, Path]] | None = self.fn_map.get(self.mode)
        if self.fn is None:
            raise KeyError(f"Unsupported mode: {self.mode}")

    def __call__(self, input_keys: List[str], input_data: Dict[str, Any]):
        if self.fn is None:
            raise ValueError("FilepathFn is not initialized.")
        return self.fn(input_keys, input_data)

    def eeg(self, input_keys: List[str], input_data: Dict[str, Any]):
        """
        "E:/Dataset/SparrKULee/sub-001/ses-shortstories01/eeg/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz"
        "E:/Dataset/SparrKULee/stimuli/eeg/audiobook_5_1.npz.gz"
        -> "E:/Dataset/SparrKULee (preprocessed)/eeg-64/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg-64.npy"
        or -> "E:/Dataset/SparrKULee (preprocessed)/eeg-512/sub-001/ses-shortstories01/sub-001_ses-shortstories01_task-listeningActive_run-01_eeg-512.npy"
        :return:
        """
        assert len(input_keys) >= 3
        o_eeg_path, o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        subject, session, task, run, ext = o_eeg_path.stem.split("_")
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[2:]:
            target_file_path = self.target_root_dir / "eeg" / subject / f"{session}_{task}_{run}" / f"{subject}_{session}_{task}_{run}_{stimuli_name}_{key}.pt"
            ret[key] = target_file_path
        return ret

    def stimulus(self, input_keys: List[str], input_data: Dict[str, Any]):
        """
        "E:/Dataset/SparrKULee/stimuli/eeg/podcast_1.npz.gz"
        -> "E:/Dataset/SparrKULee (preprocessed)/envelope/podcast_1_envelope.npy"
        -> "E:/Dataset/SparrKULee (preprocessed)/ffr/podcast_1_ffr.npy"
        """
        assert len(input_keys) >= 2
        o_stimuli_path, *_ = [input_data[k] for k in input_keys]
        stimuli_name = o_stimuli_path.stem.split(".")[0].replace("_", "-")
        ret = {}
        for key in input_keys[1:]:
            target_file_path = self.target_root_dir / "stimuli" / stimuli_name / f"{stimuli_name}_{key}.pt"
            ret[key] = target_file_path
        return ret


class Save(Step):
    def __init__(self, input_keys: Key, filepath_fn: FilepathFn, ext: str = '.pt'):
        """
        :param input_keys: recommended
        :param target_filepath_function: 用于生成目标路径的函数
        """
        super().__init__(
            input_keys,
            [],
            None,
            [DefaultKeys.RETURN_CODE]
        )
        self.assert_keys_num('>', 0, '==', 1)
        self.target_filepath_function = filepath_fn
        self.ext = ext

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        target_file_paths = self.target_filepath_function(self.input_keys, input_data)
        for key, target_file_path in target_file_paths.items():
            data = input_data[key]
            target_file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
            if self.ext:
                target_file_path = target_file_path.with_suffix(self.ext)
            if self.ext == '.pt':
                fd = os.open(target_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o777)
                with os.fdopen(fd, 'wb') as f:
                    torch.save(torch.from_numpy(data), f)
            elif self.ext == '.npy':
                np.save(target_file_path, data)
            else:
                logger.warning(f"Unsupported extension: {self.ext}, default to .pt.")
                fd = os.open(target_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o777)
                with os.fdopen(fd, 'wb') as f:
                    torch.save(torch.from_numpy(data), f)

        return {self.output_keys[0]: "success"}
