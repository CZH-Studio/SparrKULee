"""
Copy parameters from one model to another.
Example usage:
python -m match_mismatch.copy_params \
    --src_data clip-pretrain \
    --dst_data clip-finetune \
    --src_model clip-pretrain \
    --dst_model clip-finetune \
    --copy eeg_encoder,speech_encoder \
    --src_ckpt clip-pretrain_seed=1 \
    --dst_ckpt clip-finetune_seed=1
"""

from pathlib import Path
import warnings
import argparse

import yaml
import torch

from match_mismatch.model import get_litmodule, find_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data", type=str, required=True)
    parser.add_argument("--dst_data", type=str, required=True)
    parser.add_argument("--src_model", type=str, required=True)
    parser.add_argument("--dst_model", type=str, required=True)
    parser.add_argument("--src_trainer", type=str, default="default")
    parser.add_argument("--dst_trainer", type=str, default="default")
    parser.add_argument("--src_ckpt", type=str, required=True)
    parser.add_argument("--dst_ckpt", type=str, required=True)
    parser.add_argument("--copy", type=str)
    args = parser.parse_args()
    return args


def parse_params(params_str: str) -> list[str]:
    params = params_str.strip().replace(" ", "").split(",")
    return params


def copy(src_litmodule, dst_litmodule, copy_params: list[str]):
    def get_by_path(root, path):
        obj = root
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj

    for p in copy_params:
        src_module = get_by_path(src_litmodule.model, p)
        dst_module = get_by_path(dst_litmodule.model, p)
        dst_module.load_state_dict(src_module.state_dict())


def main():
    args = parse_args()
    src_data_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "data" / f"{args.src_data}.yaml",
            "r",
            encoding="utf-8",
        )
    )
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
    src_trainer_config = yaml.safe_load(
        open(
            ROOT_DIR / "config" / "trainer" / f"{args.src_trainer}.yaml",
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
    src_ckpt_path = find_ckpt(src_ckpt_dir, "best")
    dst_ckpt_path = dst_ckpt_dir / "best.ckpt"
    src_litmodule = get_litmodule(
        src_data_config, src_model_config, src_trainer_config, src_ckpt_path
    )
    dst_litmodule = get_litmodule(
        dst_data_config, dst_model_config, dst_trainer_config, None
    )
    copy_params = parse_params(args.copy)
    copy(src_litmodule, dst_litmodule, copy_params)
    torch.save(dst_litmodule, dst_ckpt_path)
    print(f"Copied parameters from {src_ckpt_path} to {dst_ckpt_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    ROOT_DIR = Path(__file__).parent
    main()
