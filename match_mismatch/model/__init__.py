from abc import ABC, abstractmethod
from typing import Optional, Type
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import importlib

from datautils.dataloader import (
    get_labels,
    get_dataloader_by_config,
    DataLoader,
    SparrKULeeDataset,
)
from datautils import Features


class MatchMismatchModel(nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, indices: torch.Tensor, x: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class MemoryBank(nn.Module):
    def __init__(
        self,
        size: int,
        embedding_dim: int,
        momentum: float = 0.9,
        device=torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.size = size
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.register_buffer(
            "memory", torch.rand(size + 1, embedding_dim, device=device)
        )

    def forward(self, indices: torch.Tensor, x: torch.Tensor):
        data_averages = torch.index_select(self.memory, 0, indices.view(-1)).detach()
        new_entry = data_averages.clone()
        with torch.no_grad():
            new_entry.mul_(self.momentum)
            new_entry.add_(torch.mul(x, 1 - self.momentum))
            self.memory.index_copy_(0, indices, new_entry)
        return data_averages


class ContrastLearningModel(nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.memory_bank: Optional[MemoryBank] = None

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        # this function implements the standard cross-entropy loss for CLIP
        # you can customize this function to use other loss functions
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
        loss = (loss_a + loss_b) / 2
        return loss

    def init_memory_bank(self, size: int, embedding_dim: int):
        return

    def get_embedding_dim(self) -> int:
        return 0

    @abstractmethod
    def forward(self, indices: torch.Tensor, x: list[torch.Tensor]):
        raise NotImplementedError()


class MyLitModule(pl.LightningModule, ABC):
    def __init__(
        self,
        data_config: dict,
        model: MatchMismatchModel | ContrastLearningModel | list[MatchMismatchModel],
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
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

    @abstractmethod
    def forward(self, batch_data: list[torch.Tensor]):
        raise NotImplementedError()

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


class LitMatchMismatchModule(MyLitModule):
    def __init__(
        self,
        data_config: dict,
        model: MatchMismatchModel | list[MatchMismatchModel],
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__(data_config, model, optimizer_config, scheduler_config)
        self.num_classes = self.data_config["num_classes"]
        self.features = Features(**self.data_config["features"])
        self.save_hyperparameters(
            ignore=["model", "criterion", "features", "num_classes"]
        )

    def forward(self, batch_data) -> tuple[torch.Tensor, float]:
        indices, *data = batch_data  # extract indices from the first position
        data, label = get_labels(self.features, self.num_classes, data)
        assert isinstance(self.model, MatchMismatchModel)
        out = self.model(indices, data)
        loss = self.criterion(out, label)
        perd = torch.argmax(out, dim=1)
        total = label.size(0)
        correct = (perd == label).sum().item()
        acc = correct / total
        return loss, acc

    def training_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss, acc = self(batch_data)
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
        loss, acc = self(batch_data)
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
        loss, acc = self(batch_data)
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


class LitMatchMismatchEnsembleModule(LitMatchMismatchModule):
    def __init__(
        self,
        data_config: dict,
        models: list[MatchMismatchModel],
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__(data_config, models, optimizer_config, scheduler_config)
        self.num_classes = data_config["num_classes"]
        self.features = Features(**data_config["features"])
        self.save_hyperparameters(
            ignore=["model", "criterion", "features", "num_classes"]
        )

    def forward(self, batch_data: list[torch.Tensor]):
        indices, *data = batch_data
        data, label = get_labels(self.features, self.num_classes, data)
        assert isinstance(self.model, list)
        outs = [model(indices, data) for model in self.model]
        out = torch.stack(outs, dim=0).mean(dim=0)
        loss = self.criterion(out, label)
        perd = torch.argmax(out, dim=1)
        total = label.size(0)
        correct = (perd == label).sum().item()
        acc = correct / total
        return loss, acc


class LitContrastLearningModule(MyLitModule):
    def __init__(
        self,
        data_config: dict,
        model: ContrastLearningModel,
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__(data_config, model, optimizer_config, scheduler_config)
        # in contrast learning, all features should be set to is_stimuli=False
        for i in range(len(self.data_config["features"]["items"])):
            self.data_config["features"]["items"][i]["is_stimuli"] = False
        self.batch_size = self.data_config["batch_size"]
        self.features = Features(**data_config["features"])
        self.save_hyperparameters(
            ignore=["model", "criterion", "features", "batch_size"]
        )

    def forward(self, batch_data: list[torch.Tensor]):
        indices, *data = batch_data
        assert isinstance(self.model, ContrastLearningModel)
        out = self.model(indices, data)
        loss = self.model.compute_loss(out)
        return loss

    def training_step(
        self, batch_data: tuple[list[torch.Tensor], torch.Tensor], batch_idx
    ) -> torch.Tensor:
        loss = self(batch_data)
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
        loss = self(batch_data)
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
        loss = self(batch_data)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_start(self) -> None:
        assert isinstance(self.trainer.train_dataloader, DataLoader)
        assert isinstance(self.trainer.train_dataloader.dataset, SparrKULeeDataset)
        assert isinstance(self.model, ContrastLearningModel)
        size = len(self.trainer.train_dataloader.dataset)
        embedding_dim = self.model.get_embedding_dim()
        self.model.init_memory_bank(size, embedding_dim)


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
            litmodule = litmodule_class.load_from_checkpoint(
                ckpt_path, **kwargs, strict=False
            )
        except TypeError:
            litmodule = torch.load(ckpt_path)
            for k, v in kwargs.items():
                if k not in banned_keys:
                    setattr(litmodule, k, v)
    return litmodule


def get_ensemble_litmodule(
    data_config: dict,
    model_config: dict,
    trainer_config: dict,
    ckpt_paths: list[Path],
) -> LitMatchMismatchEnsembleModule:
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
    models = [
        litmodule_class.load_from_checkpoint(ckpt_path, **kwargs).model
        for ckpt_path in ckpt_paths
    ]
    litensemblemodel = LitMatchMismatchEnsembleModule(
        data_config,
        models=models,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    return litensemblemodel


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
