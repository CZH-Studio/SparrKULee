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
        "eeg-64-board-band": ["envelope-64-board-band", "mel-64"],
        "eeg-512-low-gamma": ["envelope-512-board-band"],
    }
    # 每一个EEG文件都需要找到与之相匹配的刺激文件
    for subject_dir in eeg_base_dir.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue
        for eeg_dir in subject_dir.iterdir():
            if not eeg_dir.is_dir():
                continue
            example = next(eeg_dir.glob("*.pt"))
            subject, session, task, run, stimuli, feature = example.stem.split("_")
            # 找到对应的刺激文件夹
            stimuli_dir = stimuli_base_dir / stimuli
            # 加载EEG数据并处理
            for eeg_path in eeg_dir.glob("*.pt"):
                print(f"Processing {eeg_path.name}...")
                # 读取该EEG的key
                subject, session, task, run, stimuli, feature = eeg_path.stem.split("_")
                # 找到对应的刺激文件
                stimuli_features = mapping[feature]
                stimuli_paths = [
                    stimuli_dir / f"{stimuli}_{k}.pt" for k in stimuli_features
                ]
                # 读取EEG和stimuli数据
                eeg: torch.Tensor = torch.load(
                    eeg_path, weights_only=True
                )  # (n_channels, T)
                eeg = eeg.transpose(0, 1)[:, :64]  # (T, 64)
                stimuli_data: list[torch.Tensor] = [
                    torch.load(p, weights_only=True) for p in stimuli_paths
                ]  # (T, 1)
                shortest_length = min(eeg.shape[0], *[s.shape[0] for s in stimuli_data])
                # 开始划分
                eeg_mean = None
                eeg_std = None
                stimuli_mean: list = [None for _ in range(len(stimuli_features))]
                stimuli_std: list = [None for _ in range(len(stimuli_features))]
                pointer = 0
                for i, split in enumerate(splits):
                    split_name, split_fraction = map(split.get, ["split", "fraction"])
                    split_length = int(shortest_length * split_fraction)  # type: ignore
                    # 切割数据
                    eeg_split = eeg[pointer : pointer + split_length]
                    stimulus_split = [
                        s[pointer : pointer + split_length] for s in stimuli_data
                    ]
                    # 正则化
                    if eeg_mean is None:
                        eeg_mean = eeg_split.mean(dim=0)
                        eeg_std = eeg_split.std(dim=0)
                    eeg_split = (eeg_split - eeg_mean) / eeg_std
                    for j, s in enumerate(stimulus_split):
                        if stimuli_mean[j] is None:
                            stimuli_mean[j] = s.mean(dim=0)
                            stimuli_std[j] = s.std(dim=0)
                        stimulus_split[j] = (
                            stimulus_split[j] - stimuli_mean[j]
                        ) / stimuli_std[j]
                    # 保存
                    save_dir = (
                        split_dir / split_name / subject / f"{session}_{task}_{run}"  # type: ignore
                    )
                    save_dir.mkdir(exist_ok=True, parents=True, mode=0o777)
                    save_eeg_name = f"{split_name}_{subject}_{session}_{task}_{run}_{stimuli}_{feature}.pt"
                    fd = os.open(
                        save_dir / save_eeg_name,
                        os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                        0o777,
                    )
                    with os.fdopen(fd, "wb") as f:
                        torch.save(eeg_split, f)
                    for j, k in enumerate(stimuli_features):
                        save_stimuli_name = f"{split_name}_{subject}_{session}_{task}_{run}_{stimuli}_{k}.pt"
                        fd = os.open(
                            save_dir / save_stimuli_name,
                            os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                            0o777,
                        )
                        with os.fdopen(fd, "wb") as f:
                            torch.save(stimulus_split[j], f)
                        # 更新指针
                    pointer += split_length


if __name__ == "__main__":
    main()
