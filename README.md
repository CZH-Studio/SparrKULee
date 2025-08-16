# SparrKULee 数据集实验汇总

## 匹配/不匹配任务

| 任务  | 总数  | SOTA          | BMMSNet       | Baseline      |
| ----- | ----- | ------------- | ------------- | ------------- |
| 2分类 | 50048 | 43735(87.39%) | 41974(83.87%) | 41437(82.79%) |
| 5分类 | 50048 | 35150(70.23%) | 32346(64.63%) | 31073(62.09%) |

## 有关SOTA模型

只运行SOTA模型本身，参考`match_mismatch/src/experiments/sota.py`

多个SOTA并行计算，参考`match_mismatch/src/experiments/sota_parallel.py`

如果先单独训练若干个SOTA，再集成，则参考`match_mismatch/src/experiments/sota_ensemble.py`

有关SOTA模型的具体实现，参考`match_mismatch/src/models/sota.py`和`match_mismatch/src/models/dilated_conv_models.py`

所有的配置项在`match_mismatch/configs`文件夹

