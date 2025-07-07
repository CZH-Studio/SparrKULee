"""Split data in sets and normalize (per recording)."""
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    # 文件夹路径
    eeg_base_dir = Path("E:/Dataset/SparrKULee (preprocessed)/eeg")
    stimuli_base_dir = Path("E:/Dataset/SparrKULee (preprocessed)/stimuli")
    split_base_dir = Path("E:/Dataset/SparrKULee (preprocessed)/split")
    split_base_dir.mkdir(exist_ok=True, parents=True)
    # 数据集划分比例
    splits = [80, 10, 10]
    split_names = ["train", "val", "test"]
    overwrite = False
    split_fractions = [x / sum(splits) for x in splits]  # [0.8, 0.1, 0.1]
    # 一个EEG文件夹对应一个刺激文件夹，这样才能知道怎么配对
    pairs = {
        "normal": ["eeg-64", "envelope-64"],
        "high-freq": ["eeg-512", "envelope-512"],
    }
    # 开始遍历每一个配对
    for key, pair in pairs.items():
        print(f"Processing {key}...")
        eeg_type, stimuli_type = pair
        # eeg_type: eeg-64, stimuli_type: envelope-64
        eeg_dir = eeg_base_dir / eeg_type
        stimuli_dir = stimuli_base_dir / stimuli_type
        split_dir = split_base_dir / key    # E:/Dataset/SparrKULee (preprocessed)/split/normal
        split_dir.mkdir(exist_ok=True, parents=True)
        # 准备开始遍历
        for eeg_path in eeg_dir.glob("sub-*/*/*.npy"):
            print(f"Processing {eeg_path}...")
            # 从文件名读取信息
            subject, session, task, run, stimuli_name, _ = eeg_path.stem.split("_")
            # sub-001, ses-shortstories01, task-listeningActive, run-01, audiobook-1, eeg-64
            # 找到对应的刺激文件
            stimuli_path = stimuli_dir / f"{stimuli_name}_{stimuli_type}.npy"
            # 加载数据，处理
            eeg = np.load(eeg_path)             # (n_channels, T)
            stimuli = np.load(stimuli_path)                       # (T, 1)
            eeg = np.swapaxes(eeg, 0, 1)[:, :64]        # (T, 64)
            shortest_length = min(eeg.shape[0], stimuli.shape[0])
            # 开始划分
            start_index = 0
            eeg_mean = None
            eeg_std = None
            stimuli_mean = None
            stimuli_std = None
            for split_name, split_fraction in zip(split_names, split_fractions):
                end_index = start_index + int(shortest_length * split_fraction)
                # 切割数据
                eeg_split = eeg[start_index:end_index]
                stimuli_split = stimuli[start_index:end_index]
                # 正则化
                if eeg_mean is None:
                    eeg_mean = np.mean(eeg_split, axis=0)
                    eeg_std = np.std(eeg_split, axis=0)
                norm_eeg_split = (eeg_split - eeg_mean) / eeg_std
                if stimuli_mean is None:
                    stimuli_mean = np.mean(stimuli_split, axis=0)
                    stimuli_std = np.std(stimuli_split, axis=0)
                norm_stimuli_split = (stimuli_split - stimuli_mean) / stimuli_std
                # 保存
                eeg_save_filename = f"{split_name}_{subject}_{stimuli_name}_{eeg_type}.npy"
                stimuli_save_filename = f"{split_name}_{subject}_{stimuli_name}_{stimuli_type}.npy"
                np.save(split_dir / eeg_save_filename, norm_eeg_split)
                np.save(split_dir / stimuli_save_filename, norm_stimuli_split)
                # 更新索引
                start_index = end_index