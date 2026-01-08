from pathlib import Path
import argparse
from datetime import datetime
import warnings

import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from match_mismatch.model import get_litmodule, find_ckpt


def train(data_config: dict, model_config: dict, trainer_config: dict, ckpt_dir: Path):
    last_ckpt = find_ckpt(ckpt_dir, "last")
    litmodule, resume = get_litmodule(
        data_config, model_config, trainer_config, last_ckpt
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=trainer_config["early_stopping"]["min_delta"],
        patience=trainer_config["early_stopping"]["patience"],
        verbose=True,
    )
    log_dir = ROOT_DIR / "tb_logs" / ckpt_dir.name
    log_dir.mkdir(exist_ok=True, parents=True)
    tensorboard_logger = TensorBoardLogger(
        save_dir=ROOT_DIR / "tb_logs",
        name=ckpt_dir.name,
        version=None if resume else f"version-{len(list(log_dir.iterdir()))}",
    )
    trainer = pl.Trainer(
        max_epochs=trainer_config["trainer"]["max_epochs"],
        accelerator=trainer_config["trainer"]["accelerator"],
        devices=trainer_config["trainer"]["devices"],
        strategy=trainer_config["trainer"]["strategy"],
        precision=trainer_config["trainer"]["precision"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_logger,
        use_distributed_sampler=False,
    )
    if resume:
        print("Resuming from last checkpoint: ", last_ckpt)
        trainer.fit(litmodule, ckpt_path=last_ckpt)
    else:
        trainer.fit(litmodule)


def evaluate(
    data_config: dict, model_config: dict, trainer_config: dict, ckpt_dir: Path
):
    ckpt_path = find_ckpt(ckpt_dir, "best")
    litmodule, resume = get_litmodule(
        data_config, model_config, trainer_config, ckpt_path
    )
    if not resume:
        raise ValueError("No checkpoint found.")
    trainer = pl.Trainer(
        accelerator=trainer_config["trainer"]["accelerator"],
        devices=trainer_config["trainer"]["devices"],
        strategy=trainer_config["trainer"]["strategy"],
        use_distributed_sampler=False,
    )
    trainer.test(litmodule, ckpt_path=ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, required=True, help="Name of data config file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of model config file."
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="Name of trainer config file."
    )
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Whether to evaluate only."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint directory."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_config = yaml.safe_load(
        open(ROOT_DIR / "config" / "data" / f"{args.data}.yaml", "r", encoding="utf-8")
    )
    if args.seed is not None:
        data_config["seed"] = args.seed
    model_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "model" / f"{args.model}.yaml", "r", encoding="utf-8"
        )
    )
    trainer_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "trainer" / f"{args.trainer}.yaml",
            "r",
            encoding="utf-8",
        )
    )
    ckpt_dir = (
        ROOT_DIR / "ckpt" / model_config["name"] / args.ckpt
        if args.ckpt is not None
        else ROOT_DIR
        / "ckpt"
        / model_config["name"]
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    if not args.evaluate_only:
        train(data_config, model_config, trainer_config, ckpt_dir)
    evaluate(data_config, model_config, trainer_config, ckpt_dir)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    ROOT_DIR = Path(__file__).parent
    main()
