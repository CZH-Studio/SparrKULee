"""Split data in sets and normalize (per recording)."""

import os

import torch

from . import DATASET_PROCESSED_DIR


def main():
    print("Splitting and normalizing data...")
    # 文件夹路径
    eeg_base_dir = DATASET_PROCESSED_DIR / "eeg"
    stimuli_base_dir = DATASET_PROCESSED_DIR / "stimuli"
    split_dir = DATASET_PROCESSED_DIR / "split"
    split_dir.mkdir(exist_ok=True, parents=True, mode=0o777)
    # 数据集划分比例
    splits = [
        {"split": "train", "fraction": 80},
        {"split": "val", "fraction": 10},
        {"split": "test", "fraction": 10},
    ]
    total: int | float = sum(s["fraction"] for s in splits)
    for s in splits:
        fraction: int | float = s["fraction"]
        s["fraction"] = fraction / total  # Noqa
    # eeg存放在 root/eeg/sub-001/{session}_{task}_{run}/{subject}_{session}_{task}_{run}_{stimuli_name}_{key}.pt
    # stimuli存放在 root/stimuli/{stimuli_name}/{stimuli_name}_{key}.pt
    # 为每一个eeg文件的key设置与之相对应的stimuli文件的key
    mapping = {
        "eeg-64-board-band": "envelope-64-board-band",
        "eeg-512-low-gamma": "envelope-512-board-band",
    }
    # 每一个EEG文件都需要找到与之相匹配的刺激文件
    for subject_dir in eeg_base_dir.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue
        for eeg_dir in subject_dir.iterdir():
            if not eeg_dir.is_dir():
                continue
            example = next(eeg_dir.glob("*.pt"))
            subject, session, task, run, stimuli_name, key = example.stem.split("_")
            # 找到对应的刺激文件夹
            stimuli_dir = stimuli_base_dir / stimuli_name
            # 加载EEG数据并处理
            for eeg_path in eeg_dir.glob("*.pt"):
                print(f"Processing {eeg_path.name}...")
                # 读取该EEG的key
                subject, session, task, run, stimuli_name, key = eeg_path.stem.split(
                    "_"
                )
                # 找到对应的刺激文件
                stimuli_path = stimuli_dir / f"{stimuli_name}_{mapping[key]}.pt"
                # 读取EEG和stimuli数据
                eeg: torch.Tensor = torch.load(
                    eeg_path, weights_only=True
                )  # (n_channels, T)
                eeg = eeg.transpose(0, 1)[:, :64]  # (T, 64)
                stimuli = torch.load(stimuli_path, weights_only=True)  # (T, 1)
                shortest_length = min(eeg.shape[0], stimuli.shape[0])
                # 开始划分
                eeg_mean, eeg_std, stimuli_mean, stimuli_std = None, None, None, None
                pointer = 0
                for i, split in enumerate(splits):
                    split_name, split_fraction = map(split.get, ["split", "fraction"])
                    assert isinstance(split_name, str) and isinstance(
                        split_fraction, int
                    )
                    split_length = int(shortest_length * split_fraction)
                    # 切割数据
                    eeg_split = eeg[pointer : pointer + split_length]
                    stimuli_split = stimuli[pointer : pointer + split_length]
                    # 正则化
                    if eeg_mean is None:
                        eeg_mean = eeg_split.mean(dim=0)
                        eeg_std = eeg_split.std(dim=0)
                    norm_eeg_split = (eeg_split - eeg_mean) / eeg_std
                    if stimuli_mean is None:
                        stimuli_mean = stimuli_split.mean(dim=0)
                        stimuli_std = stimuli_split.std(dim=0)
                    norm_stimuli_split = (stimuli_split - stimuli_mean) / stimuli_std
                    # 保存
                    save_dir = (
                        split_dir / split_name / subject / f"{session}_{task}_{run}"
                    )
                    save_dir.mkdir(exist_ok=True, parents=True, mode=0o777)
                    save_eeg_name = f"{split_name}_{subject}_{session}_{task}_{run}_{stimuli_name}_{key}.pt"
                    save_stimuli_name = f"{split_name}_{subject}_{session}_{task}_{run}_{stimuli_name}_{mapping[key]}.pt"
                    fd = os.open(
                        save_dir / save_eeg_name,
                        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                        0o777,
                    )
                    with os.fdopen(fd, "wb") as f:
                        torch.save(norm_eeg_split, f)
                    fd = os.open(
                        save_dir / save_stimuli_name,
                        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                        0o777,
                    )
                    with os.fdopen(fd, "wb") as f:
                        torch.save(norm_stimuli_split, f)
                    # 更新指针
                    pointer += split_length


if __name__ == "__main__":
    main()
