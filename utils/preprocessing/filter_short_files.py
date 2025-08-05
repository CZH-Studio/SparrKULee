# 有些文件的长度比较短，无法采样到足够的不匹配段，因此可以将这些文件过滤掉，放到其他文件夹中避免使用
from pathlib import Path
import math
import shutil

import torch

input_dir = Path(r'E:\Dataset\SparrKULee (preprocessed)\split')
output_dir = Path(r'E:\Dataset\SparrKULee (preprocessed)\split (deprecated)')
output_dir.mkdir(exist_ok=True, parents=True)
window_length_t = 5
step_length_t = 1
min_spacing_length_t = 6

for file in input_dir.iterdir():
    if file.is_file():
        _, _, _, feature = file.stem.split('_')
        sample_rate = int(feature.split('-')[1])
        window_length = int(window_length_t * sample_rate)
        step_length = int(step_length_t * sample_rate)
        min_spacing_length = int(min_spacing_length_t * sample_rate)
        data = torch.load(file, weights_only=True)
        t = data.shape[0]
        l_min = math.ceil(min_spacing_length / step_length) * step_length + window_length
        if t < l_min:
            print(f'{file.stem}: {t} < {l_min}, filtered.')
            dst = output_dir / file.name
            shutil.move(str(file), str(dst))
