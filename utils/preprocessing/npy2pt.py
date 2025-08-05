# 将npy文件转换为torch.Tensor(pt)文件
from pathlib import Path

import numpy as np
import torch

path = Path(r'E:\Dataset\SparrKULee (preprocessed)\split')
for file in path.iterdir():
    if file.is_file() and file.suffix == '.npy':
        tensor = torch.from_numpy(np.load(file))
        save_path = file.parent / f'{file.stem}.pt'
        torch.save(tensor, save_path)
        file.unlink()
