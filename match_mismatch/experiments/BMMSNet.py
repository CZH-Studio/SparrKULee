from pathlib import Path

import torch

from match_mismatch.models.BMMSNet.dense_net import BMMSNet
from match_mismatch.utils.config import Config, parse_args
from match_mismatch.utils.tve import train, evaluate


def main():
    # Parameters
    args = parse_args()
    config = Config(
        data_dir=Path("E:/Dataset/SparrKULee (preprocessed)/split"),
        executing_file_path=Path(__file__),
        selected_features={
            "normal": ({"eeg-64": False, "envelope-64": True}, 64),
        },
        evaluate_only=args.evaluate_only,
        preserve=args.preserve,
        num_classes=args.num_classes,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
    )

    if config.evaluate_only:
        if not config.model_path.exists():
            raise FileNotFoundError("Model file does not exist.")
        model = torch.load(config.model_path, weights_only=False)
    else:
        model = BMMSNet()
        model = train(model, config)
    evaluate(model, config)


if __name__ == "__main__":
    main()
