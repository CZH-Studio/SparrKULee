from datetime import datetime
from pathlib import Path
from typing import Any, Type, List, Tuple
import copy

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from omegaconf import DictConfig
from tqdm import tqdm

from match_mismatch.src.data.dataset import SparrKULeeDatasetManager
from match_mismatch.src.models import get_model

_INIT_ = False

def _init_(cfg, results_dir):
    global _INIT_
    if _INIT_:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')
    _INIT_ = True


def rename_model_name(cfg, new_name):
    """
    返回一个全新的cfg，其model.model_name属性修改为new_name
    :param cfg: dict config
    :param new_name: 名称
    :return: cfg_new
    """
    cfg_new = copy.deepcopy(cfg)
    cfg_new.model.model_name = new_name
    return cfg_new


class DynamicProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self._total_train_batches = 0
        self._total_val_batches = 0
        self.is_total_train_unknown = True
        self.is_total_val_unknown = True

    def init_train_epoch_tqdm(self, epoch: int):
        return tqdm(
            total=self._total_train_batches,
            desc=f"{self.train_description} Epoch {epoch + 1}",
            unit="batch",
        )

    def init_val_epoch_tqdm(self, epoch: int):
        return tqdm(
            total=self._total_val_batches,
            desc=f"{self.validation_description} Epoch {epoch + 1}",
            unit="batch",
        )

    def init_test_epoch_tqdm(self):
        return tqdm(
            total=0,
            desc=f"{self.test_description}",
            unit="batch",
        )

    def on_train_start(self, *_: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.train_progress_bar = self.init_train_epoch_tqdm(trainer.current_epoch)

    def on_train_batch_end(self, trainer: "pl.Trainer", *_) -> None:
        self.train_progress_bar.update(1)
        if self.is_total_train_unknown:
            self._total_train_batches += 1

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # 在验证开始前关闭训练的进度条
        self.is_total_train_unknown = False
        self.train_progress_bar.close()

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_progress_bar = self.init_val_epoch_tqdm(trainer.current_epoch)

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.val_progress_bar.update(1)
        if self.is_total_val_unknown:
            self._total_val_batches += 1

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.is_total_val_unknown = False
        self.val_progress_bar.close()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar = self.init_test_epoch_tqdm()

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.test_progress_bar.update(1)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_progress_bar.close()


class LitBaseModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.stats = torch.zeros((3, 2), dtype=torch.float32)  # 3行2列的矩阵，分别记录训练集、验证集、测试集的total、correct

    def calc_step(self, batch_data: Tuple[List[torch.Tensor], torch.Tensor]):
        data, label = batch_data
        out = self(data)
        loss = self.criterion(out, label)
        pred = torch.argmax(out, dim=1)
        total = len(label)
        correct = (pred == label).sum().item()
        return loss, total, correct

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch_data, batch_idx):
        loss, total, correct = self.calc_step(batch_data)
        self.stats[0, 0] += total
        self.stats[0, 1] += correct
        self.log('train_loss', loss, on_step=True)
        return loss

    def on_train_epoch_start(self) -> None:
        self.stats[0, 0] = 0
        self.stats[0, 1] = 0

    def on_train_epoch_end(self) -> None:
        acc = self.stats[0, 1] / self.stats[0, 0] if self.stats[0, 0] > 0 else 0.0
        self.log('train_acc', acc, on_epoch=True)

    def validation_step(self, batch_data, batch_idx):
        loss, total, correct = self.calc_step(batch_data)
        self.stats[1, 0] += total
        self.stats[1, 1] += correct
        self.log('val_loss', loss, on_step=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.stats[1, 0] = 0
        self.stats[1, 1] = 0

    def on_validation_epoch_end(self) -> None:
        acc = self.stats[1, 1] / self.stats[1, 0] if self.stats[1, 0] > 0 else 0.0
        self.log('val_acc', acc, on_epoch=True)

    def test_step(self, batch_data, batch_idx):
        loss, total, correct = self.calc_step(batch_data)
        self.stats[2, 0] += total
        self.stats[2, 1] += correct
        self.log('test_loss', loss, on_step=True)

    def on_test_epoch_start(self) -> None:
        self.stats[2, 0] = 0
        self.stats[2, 1] = 0

    def on_test_epoch_end(self) -> None:
        total = self.stats[2, 0]
        correct = self.stats[2, 1]
        acc = correct / total if total > 0 else 0.0
        self.log('test_acc', acc, on_epoch=True)
        self.log('test_correct', correct, on_epoch=True)
        self.log('test_total', total, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.cfg.trainer.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }


def train(model_class: "Type[pl.LightningModule]", cfg: DictConfig, results_dir: Path):
    _init_(cfg, results_dir)
    ckpt_dir = results_dir / "models"
    model_path = ckpt_dir / (cfg.model.save_name + ".ckpt")
    model = model_class(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=cfg.model.save_name,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=cfg.trainer.early_stopping_patience,
        mode="max",
        verbose=True,
    )
    logger = TensorBoardLogger(
        save_dir=str(results_dir),
        name="logs",
        version=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    progressbar = DynamicProgressBar()
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        callbacks=[checkpoint_callback, early_stopping_callback, progressbar],
        logger=logger,
        num_sanity_val_steps=0
    )
    if isinstance(cfg.data.num_subjects, int):
        subjects = list(range(1, cfg.data.num_subjects + 1))
    elif isinstance(cfg.data.num_subjects, list):
        subjects = cfg.data.num_subjects
    else:
        raise ValueError("num_subjects should be int or list")
    dataset_manager = SparrKULeeDatasetManager(data_dir=cfg.data.data_dir)
    train_loader = dataset_manager.match_mismatch_dataloader(
        "train", subjects=subjects, batch_size=cfg.trainer.batch_size, **cfg.data
    )
    val_loader = dataset_manager.match_mismatch_dataloader(
        "val", subjects=subjects, batch_size=cfg.trainer.batch_size, **cfg.data
    )
    if model_path.exists() and cfg.continue_training:
        trainer.fit(model, train_loader, val_loader, ckpt_path=model_path)
    else:
        trainer.fit(model, train_loader, val_loader)
    best_val_acc = checkpoint_callback.best_model_score.item()
    return trainer, model, best_val_acc


def load(model_class: "Type[pl.LightningModule]", cfg: DictConfig, results_dir: Path):
    _init_(cfg, results_dir)
    model_path = results_dir / "models" / (cfg.model.save_name + ".ckpt")
    model = model_class.load_from_checkpoint(model_path, cfg=cfg)
    progressbar = DynamicProgressBar()
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        callbacks=[progressbar],
        logger=False,
    )
    return trainer, model


def save(model_class: "Type[pl.LightningModule]", cfg: DictConfig, results_dir: Path):
    _init_(cfg, results_dir)
    model = model_class(cfg)
    model_path = results_dir / "models" / (cfg.model.save_name + ".ckpt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        'pytorch-lightning_version': pl.__version__,
    }, model_path)


def test(cfg, results_dir, trainer, model):
    _init_(cfg, results_dir)
    df = pd.DataFrame(columns=["test_acc", "test_loss", "test_total", "test_correct"])
    if isinstance(cfg.data.num_subjects, int):
        subjects = list(range(1, cfg.data.num_subjects + 1))
    elif isinstance(cfg.data.num_subjects, list):
        subjects = cfg.data.num_subjects
    else:
        raise ValueError("num_subjects should be int or list")
    dataset_manager = SparrKULeeDatasetManager(data_dir=cfg.data.data_dir)
    for subject in subjects:
        test_loader = dataset_manager.match_mismatch_dataloader(
            "test", subjects=[subject], batch_size=cfg.trainer.batch_size, **cfg.data
        )
        trainer.test(model, test_loader)
        results = trainer.callback_metrics
        try:
            df.loc[subject] = [results[key].item() for key in df.columns]
        except KeyError:
            df.loc[subject] = [0.0, 0.0, 0, 0]

    total = df["test_total"].sum()
    correct = df["test_correct"].sum()
    acc = correct / total if total > 0 else 0.0
    loss = df["test_loss"].mean()
    df.loc["Average"] = [acc, loss, total, correct]
    # 转换显示格式
    df["test_acc"] = df["test_acc"].apply(lambda x: f'{x:.2%}')
    df["test_loss"] = df["test_loss"].apply(lambda x: f'{x:.4f}')
    df["test_total"] = df["test_total"].astype(int)
    df["test_correct"] = df["test_correct"].astype(int)
    # 保存
    df.to_csv(results_dir / "test_results.csv", index_label="Subject")


def run(model_class: "Type[pl.LightningModule]", cfg, results_dir, immediate_save=False):
    # match_mismatch / output / baseline / default_experiment
    _init_(cfg, results_dir)

    if not cfg.test_only:
        train(model_class, cfg, results_dir)
    if immediate_save:
        # 该选项用于跳过了训练过程，但是希望先保存一下模型，以便后续测试用
        save(model_class, cfg, results_dir)
    trainer, model = load(model_class, cfg, results_dir)
    test(cfg, results_dir, trainer, model)
