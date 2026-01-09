"""
Copy parameters from one model to another.
Example usage:
python -m match_mismatch.copy_params \
    --dst_data cl \
    --dst_model cl-cls \
    --copy eeg_encoder,speech_encoder \
    --src_ckpt clip_seed=1 \
    --dst_ckpt clip-cls_seed=1
"""

from pathlib import Path
import warnings
import argparse
from typing import Any

import yaml
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from match_mismatch.model import get_litmodule, find_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst_data", type=str, required=True)
    parser.add_argument("--dst_trainer", type=str, default="default")
    parser.add_argument("--src_model", type=str, required=True)
    parser.add_argument("--dst_model", type=str, required=True)
    parser.add_argument("--src_ckpt", type=str, required=True)
    parser.add_argument("--dst_ckpt", type=str, required=True)
    parser.add_argument("--copy", type=str)
    args = parser.parse_args()
    return args


def parse_params(params_str: str) -> list[str]:
    params = params_str.strip().replace(" ", "").split(",")
    return params


def copy(src_litmodule, dst_litmodule: pl.LightningModule, copy_params: list[str]):
    def get_dict(root: dict[str, Any], path):
        path = "model." + path
        # 获取模型的state_dict，这行代码的含义是：
        # 找到所有以model.path开头的值，并返回以path后面内容为key的dict
        # 例如：model.encoder.conv_0 -> {conv_0: value}
        model_state = {
            k[len(path) + 1 :]: v for k, v in root.items() if k.startswith(path)
        }
        return model_state

    def get_by_path(root, path):
        obj = root
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj

    for p in copy_params:
        src_module = get_dict(src_litmodule["state_dict"], p)
        dst_module = get_by_path(dst_litmodule.model, p)
        missing, unexpected = dst_module.load_state_dict(src_module, strict=False)
        if missing:
            warnings.warn(f"Missing keys: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys: {unexpected}")


def main():
    args = parse_args()
    dst_data_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "data" / f"{args.dst_data}.yaml",
            "r",
            encoding="utf-8",
        )
    )
    src_model_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "model" / f"{args.src_model}.yaml",
            "r",
            encoding="utf-8",
        )
    )
    dst_model_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "model" / f"{args.dst_model}.yaml",
            "r",
            encoding="utf-8",
        )
    )
    dst_trainer_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "trainer" / f"{args.dst_trainer}.yaml",
            "r",
            encoding="utf-8",
        )
    )
    src_ckpt_dir = ROOT_DIR / "ckpt" / src_model_config["name"] / args.src_ckpt
    dst_ckpt_dir = ROOT_DIR / "ckpt" / dst_model_config["name"] / args.dst_ckpt
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
    src_ckpt_path = find_ckpt(src_ckpt_dir, "last")
    dst_ckpt_path = dst_ckpt_dir / "best.ckpt"
    if src_ckpt_path is None:
        raise ValueError(f"No checkpoint found in {src_ckpt_dir}")
    if dst_ckpt_path.exists():
        warnings.warn(f"{dst_ckpt_path} already exists, overriding...")
    src_litmodule = torch.load(src_ckpt_path, weights_only=True, map_location="cpu")
    dst_litmodule, _ = get_litmodule(
        dst_data_config, dst_model_config, dst_trainer_config, None
    )  # init a raw module
    copy_param_names = parse_params(args.copy)
    copy(src_litmodule, dst_litmodule, copy_param_names)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=dst_ckpt_dir,
        filename="best",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=dst_trainer_config["early_stopping"]["min_delta"],
        patience=dst_trainer_config["early_stopping"]["patience"],
        verbose=True,
    )
    log_dir = ROOT_DIR / "tb_logs" / dst_ckpt_dir.name
    log_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_logger = TensorBoardLogger(
        save_dir=ROOT_DIR / "tb_logs",
        name=dst_ckpt_dir.name,
        version=f"version-{len(list(log_dir.iterdir()))}",
    )
    trainer = pl.Trainer(
        max_epochs=dst_trainer_config["trainer"]["max_epochs"],
        accelerator=dst_trainer_config["trainer"]["accelerator"],
        devices=dst_trainer_config["trainer"]["devices"],
        strategy=dst_trainer_config["trainer"]["strategy"],
        precision=dst_trainer_config["trainer"]["precision"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        use_distributed_sampler=False,
    )
    trainer.test(dst_litmodule)
    trainer.save_checkpoint(dst_ckpt_path)
    print(f"Copied parameters from {src_ckpt_path} to {dst_ckpt_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    ROOT_DIR = Path(__file__).parent
    main()
