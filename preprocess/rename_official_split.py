# 官方提供的已经预处理后并分割好的split_data文件夹中，其文件名格式为
# test_-_sub-001_-_audiobook_1_-_eeg.npy
# 本脚本的作用是将文件名转换为统一的格式，并转换为tensor，即
# test/sub-001/audiobook_1/test_sub-001_audiobook-1_eeg.pt
# 在使用时，只需要用到test、sub-001和特征名（eeg），其他的部分无所谓
import torch
import numpy as np

from . import DATASET_RAW_SPLIT_DIR

for split in ["train", "val", "test"]:
    (DATASET_RAW_SPLIT_DIR / split).mkdir(exist_ok=True, parents=True)
for file in DATASET_RAW_SPLIT_DIR.glob("*.npy"):
    split, subject, record, feature = file.stem.split("_-_")
    record = record.replace("_", "-")
    record_dir = DATASET_RAW_SPLIT_DIR / split / subject / record
    record_dir.mkdir(exist_ok=True, parents=True)
    data = np.load(file)
    tensor = torch.from_numpy(data)
    torch.save(tensor, record_dir / f"{split}_{subject}_{record}_{feature}.pt")
    file.unlink()
