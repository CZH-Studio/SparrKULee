import torch

def vectorized_swap(cut_stimuli: torch.Tensor, labels: torch.Tensor):
    B, T, C = cut_stimuli.shape

    # 构造交换索引
    batch_indices = torch.arange(B)

    # 获取第一列的值和目标列的值
    col0 = cut_stimuli[:, :, 0].clone()                       # shape: [B, T]
    target_col = cut_stimuli[batch_indices, :, labels]        # shape: [B, T]

    # 创建副本进行交换
    swapped = cut_stimuli.clone()

    # 执行交换
    swapped[:, :, 0] = target_col
    swapped[batch_indices, :, labels] = col0

    return swapped

cut_stimuli = torch.arange(3 * 5 * 4).reshape(3, 5, 4)

labels = torch.tensor([0, 1, 3])

out = vectorized_swap(cut_stimuli, labels)
print(out)