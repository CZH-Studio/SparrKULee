import csv
import time

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from typing import List, Union

from match_mismatch.models.basemodel import MatchMismatchModel
from match_mismatch.utils.config import Config
from match_mismatch.utils.dataset import dataloader


def train(model: MatchMismatchModel, config: Config):
    train_start_time = time.time()
    # 数据集
    train_loader = dataloader(
        config.data_dir, "train", config.selected_features, config.window_length, config.step_length,
        config.spacing_length, config.num_classes, config.preserve, -1, config.batch_size,
    )
    val_loader = dataloader(
        config.data_dir, "val", config.selected_features, config.window_length, config.step_length,
        config.spacing_length, config.num_classes, config.preserve, -1, config.batch_size,
    )

    # 初始化日志文件
    is_log_exists = config.train_log_path.exists()
    if is_log_exists:
        config.train_log_path.unlink()
    with open(config.train_log_path, "w") as f:
        f.write(
            "epoch,t_acc,t_loss,t_total,t_correct,v_acc,v_loss,v_total,v_correct,time\n"
        )

    # 优化器、损失函数
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    patience = 0
    best_val_acc = 0.0
    train_loader_length = 0
    train_loader_length_exists = False
    val_loader_length = 0
    val_loader_length_exists = False
    model.to(config.device)
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        # train
        model.train()
        t_total = 0
        t_correct = 0
        running_loss = 0.0
        for data, label in tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}",
            total=None if not train_loader_length_exists else train_loader_length,
            unit="batches",
        ):
            data: List[torch.Tensor] = [d.to(config.device) for d in data]
            label: torch.Tensor = label.to(config.device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out: torch.Tensor = model(data)
                loss = criterion(out, label)
            t_total += label.size(0)
            t_correct += (out.argmax(dim=1) == label).sum().item()
            running_loss += loss.item() * label.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # 记录数据集长度
            if not train_loader_length_exists:
                train_loader_length += 1
        train_acc = t_correct / t_total if t_total > 0 else 0.0
        train_loss = running_loss / t_total if t_total > 0 else 0.0
        print(f"Train accuracy: {train_acc:.2%}, loss: {train_loss:.4f}")
        train_loader_length_exists = True

        # Validation
        model.eval()
        with torch.no_grad():
            v_total = 0
            v_correct = 0
            running_loss = 0.0
            for data, label in tqdm(
                val_loader,
                desc=f"Val Epoch {epoch}",
                total=None if not val_loader_length_exists else val_loader_length,
                unit="batches",
            ):
                data: List[torch.Tensor] = [d.to(config.device) for d in data]
                label: torch.Tensor = label.to(config.device)
                with torch.amp.autocast('cuda'):
                    out: torch.Tensor = model(data)
                    loss = criterion(out, label)
                running_loss += loss.item() * label.size(0)
                v_total += label.size(0)
                v_correct += (out.argmax(dim=1) == label).sum().item()
                # 记录数据集长度
                if not val_loader_length_exists:
                    val_loader_length += 1
            val_acc = v_correct / v_total if v_total > 0 else 0.0
            val_loss = running_loss / v_total if v_total > 0 else 0.0
            print(f"Val accuracy: {val_acc:.2%}, loss: {val_loss:.4f}")
            val_loader_length_exists = True

        epoch_end_time = time.time()
        # log and save best model
        with open(config.train_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f'{train_acc:.2%}',
                    f'{train_loss:.4f}',
                    t_total,
                    t_correct,
                    f'{val_acc:.2%}',
                    f'{val_loss:.4f}',
                    v_total,
                    v_correct,
                    second2time(epoch_end_time - epoch_start_time),
                ]
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model, config.model_path)
            print(f"Best model saved at epoch {epoch}")
        else:
            patience += 1
            print(f"Patience: {patience}/{config.patience}")
            if patience >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_end_time = time.time()
    print(
        f"Training finished. Time costs: {second2time(train_end_time - train_start_time)}"
    )
    return model


def evaluate(model: MatchMismatchModel, config: Config):
    eval_start_time = time.time()
    model.eval()
    statistics = pd.DataFrame(columns=['e_acc', 'e_loss', 'e_total', 'e_correct'])
    for subject_no in range(1, config.num_subjects + 1):
        test_loader = dataloader(
            config.data_dir, "test", config.selected_features, config.window_length, config.step_length,
            config.spacing_length, config.num_classes, config.preserve, subject_no, config.batch_size,
        )
        total = 0
        correct = 0
        loss_value = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, label in tqdm(
                test_loader, desc=f"Test Subject {subject_no}", unit="batches"
            ):
                data: List[torch.Tensor] = [d.to(config.device) for d in data]
                label: torch.Tensor = label.to(config.device)
                with torch.amp.autocast('cuda'):
                    out: torch.Tensor = model(data)
                    loss = criterion(out, label)
                total += label.size(0)
                correct += (out.argmax(dim=1) == label).sum().item()
                loss_value += loss.item() * label.size(0)
            test_acc = correct / total if total > 0 else 0.0
            loss_value = loss_value / total if total > 0 else 0.0
            statistics.loc[subject_no] = [test_acc, loss_value, total, correct]
    all_total = statistics['e_total'].sum()
    all_correct = statistics['e_correct'].sum()
    all_acc = all_correct / all_total if all_total > 0 else 0.0
    all_loss = (statistics['e_loss'] * statistics['e_total']).sum() / all_total if all_total > 0 else 0.0
    statistics.loc["Average"] = [all_acc, all_loss, all_total, all_correct]
    statistics['e_acc'] = statistics['e_acc'].apply(lambda x: f'{x:.2%}')
    statistics['e_loss'] = statistics['e_loss'].apply(lambda x: f'{x:.4f}')
    statistics['e_total'] = statistics['e_total'].astype(int)
    statistics['e_correct'] = statistics['e_correct'].astype(int)
    statistics.to_csv(config.eval_results_path, index_label="subject")
    eval_end_time = time.time()
    print(
        f"Evaluation finished, results saved to {config.eval_results_path}\n"
        f"Time costs: {second2time(eval_end_time - eval_start_time)}"
    )


def second2time(seconds: Union[int, float]) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d == 0:
        if h == 0:
            if m == 0:
                return f"{s:d}s"
            else:
                return f"{m:d}min{s:02d}s"
        else:
            return f"{h:d}h{m:02d}min{s:02d}s"
    else:
        return f"{d:d}d{h:02d}h{m:02d}min{s:02d}s"
