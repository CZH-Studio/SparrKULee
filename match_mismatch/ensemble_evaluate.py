from pathlib import Path
import argparse
from datetime import datetime
import warnings

import yaml
import lightning.pytorch as pl

from match_mismatch.model import get_litensemblemodule, find_ckpt


def evaluate(
    data_config: dict, model_config: dict, trainer_config: dict, ckpt_dirs: list[Path]
):
    pl.seed_everything(data_config["seed"])
    ckpt_paths = [find_ckpt(ckpt_dir, "best") for ckpt_dir in ckpt_dirs]
    if not all(ckpt_paths):
        raise ValueError("Some checkpoints are missing.")
    litmodule = get_litensemblemodule(
        data_config, model_config, trainer_config, ckpt_paths
    )
    trainer = pl.Trainer(
        accelerator=trainer_config["trainer"]["accelerator"],
        devices=trainer_config["trainer"]["devices"],
        strategy=trainer_config["trainer"]["strategy"],
        use_distributed_sampler=False,
    )
    trainer.test(litmodule)


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
        "--ckpt", type=str, default=None, help="Path to checkpoint directory."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
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
    ckpt_dirs = [
        ROOT_DIR / "ckpt" / model_config["name"] / ckpt for ckpt in args.ckpt.split(",")
    ]
    evaluate(data_config, model_config, trainer_config, ckpt_dirs)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    ROOT_DIR = Path(__file__).parent
    main()
