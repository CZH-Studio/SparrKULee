from abc import ABC, abstractmethod
from typing import Type
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import importlib

from datautils.dataloader import get_labels, get_dataloader_by_config
from datautils import Feature


class MatchMismatchModel(nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class ContrastLearningModel(nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # this function implements the standard cross-entropy loss for CLIP
        # you can customize this function to use other loss functions
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
        loss = (loss_a + loss_b) / 2
        return loss

    @abstractmethod
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class LitContrastLearningModule(pl.LightningModule):
    def __init__(
        self,
        data_config: dict,
        model: ContrastLearningModel,
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.batch_size = data_config["batch_size"]
        self.features = get_features(data_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=["model", "features", "batch_size"])
        self.model = model

    def train_dataloader(self):
        return get_dataloader_by_config(
            "train",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def val_dataloader(self):
        return get_dataloader_by_config(
            "val",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def test_dataloader(self):
        return get_dataloader_by_config(
            "test",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def forward(self, x: list[torch.Tensor]):
        return self.model(x)

    def _compute(self, batch_data) -> torch.Tensor:
        # in contrast learning, accuracy is not as important as loss, so we don't compute accuracy here
        out = self(batch_data)
        loss = self.model.compute_loss(out)
        return loss

    def training_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss = self._compute(batch_data)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss = self._compute(batch_data)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def test_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss = self._compute(batch_data)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):  # type: ignore
        optimizer: torch.optim.Optimizer = getattr(
            torch.optim, self.optimizer_config["name"]
        )(self.parameters(), **self.optimizer_config["params"])
        scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(
            torch.optim.lr_scheduler, self.scheduler_config["name"]
        )(optimizer, **self.scheduler_config["params"])
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        return ret


class LitMatchMismatchModule(pl.LightningModule):
    def __init__(
        self,
        data_config: dict,
        model: MatchMismatchModel,
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.num_classes = data_config["num_classes"]
        self.features = get_features(data_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=["model", "features", "num_classes"])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self):
        return get_dataloader_by_config(
            "train",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def val_dataloader(self):
        return get_dataloader_by_config(
            "val",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def test_dataloader(self):
        return get_dataloader_by_config(
            "test",
            **self.data_config,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

    def forward(self, x: list[torch.Tensor]):
        return self.model(x)

    def _compute(self, batch_data) -> tuple[torch.Tensor, float]:
        data, label = get_labels(self.features, self.num_classes, batch_data)
        out = self(data)
        loss = self.criterion(out, label)
        perd = torch.argmax(out, dim=1)
        total = label.size(0)
        correct = (perd == label).sum().item()
        acc = correct / total
        return loss, acc

    def training_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss, acc = self._compute(batch_data)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss, acc = self._compute(batch_data)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_acc = self.trainer.callback_metrics.get("val_acc", 0)
        self.print(f"Epoch {self.current_epoch}: val_acc: {val_acc:.2%}")

    def test_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss, acc = self._compute(batch_data)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("test_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        optimizer: torch.optim.Optimizer = getattr(
            torch.optim, self.optimizer_config["name"]
        )(self.parameters(), **self.optimizer_config["params"])
        scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(
            torch.optim.lr_scheduler, self.scheduler_config["name"]
        )(optimizer, **self.scheduler_config["params"])
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        return ret


def get_features(data_config: dict) -> list[Feature]:
    return [
        Feature(
            name=f["name"],
            sr=f["sr"],
            is_stimuli=f["is_stimuli"],
            random_strategy=f.get("random_strategy", "random"),
        )
        for f in data_config["features"]
    ]


def get_model_class(
    model_name: str,
) -> Type[MatchMismatchModel | ContrastLearningModel]:
    module_name, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module("match_mismatch.model." + module_name)
    return getattr(module, class_name)


def get_model(model_config: dict) -> MatchMismatchModel | ContrastLearningModel:
    model_class = get_model_class(model_config["class"])
    model = model_class(**model_config["params"])
    return model


def get_litmodule(
    data_config: dict,
    model_config: dict,
    trainer_config: dict,
    ckpt_path: Path | None = None,
) -> LitMatchMismatchModule | LitContrastLearningModule:
    """获取lit module，当指定ckpt_path时，加载ckpt，否则初始化"""
    model = get_model(model_config)
    optimizer_config = trainer_config["optimizer"]
    scheduler_config = trainer_config["scheduler"]
    kwargs = {
        "data_config": data_config,
        "model": model,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
    }
    model_module_mapping = {
        "MatchMismatchModel": LitMatchMismatchModule,
        "ContrastLearningModel": LitContrastLearningModule,
    }
    model_class = model.__class__.__bases__[0].__name__
    litmodule_class = model_module_mapping[model_class]
    banned_keys = set(["model"])

    if ckpt_path is None or not ckpt_path.exists():
        litmodule = litmodule_class(**kwargs)
    else:
        try:
            litmodule = litmodule_class.load_from_checkpoint(ckpt_path, **kwargs)
        except TypeError:
            litmodule = torch.load(ckpt_path)
            for k, v in kwargs.items():
                if k not in banned_keys:
                    setattr(litmodule, k, v)
    return litmodule


def find_ckpt(ckpt_dir: Path, mode="last"):
    """判断当前路径下是否存在最后保存/最好的模型

    Args:
        ckpt_dir (Path): 模型跟路径

    Returns:
        Path | None: 模型路径
    """
    if not ckpt_dir.exists():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"
    if mode == "last":
        if last_ckpt.exists():
            return last_ckpt
        elif best_ckpt.exists():
            return best_ckpt
        else:
            return None
    elif mode == "best":
        if best_ckpt.exists():
            return best_ckpt
        elif last_ckpt.exists():
            return last_ckpt
        else:
            return None
    else:
        raise ValueError("mode must be 'last' or 'best'")
