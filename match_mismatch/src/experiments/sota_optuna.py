import copy

import hydra
import optuna
import torch

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, train


class LitSOTAModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def calc_step(self, batch_data):
        data, label = batch_data
        out = self(data)
        loss = self.criterion(out, label)
        pred = torch.argmax(out, dim=1)
        total = len(label)
        correct = (pred == label).sum().item()
        return loss, total, correct


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    def objective(trial: optuna.Trial):
        cfg_trial = copy.deepcopy(cfg)

        # 设置需要微调的超参数
        cfg_trial.trainer.lr = trial.suggest_float("trainer.lr", 1e-4, 1e-2, log=True)
        cfg_trial.model.sota.hidden_dim = trial.suggest_int("model.sota.hidden_dim", 8, 20)

        trainer, model, best_val_acc = train(LitSOTAModule, cfg_trial, results_dir)
        if best_val_acc is None:
            return 0.0
        return best_val_acc

    # 手动修改选择的数据，以及使用的模型
    cfg.data.dataloader.selected_features = {
        "train": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, },
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        },
        "val": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, },
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        },
        "test": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, },
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        }
    }
    cfg.model.model_name = "sota"
    cfg.data.num_subjects = 20

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    results_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="sota_optuna",
        storage=f"sqlite:///match_mismatch/output/{current_file_path.stem}/{current_file_path.stem}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=30)
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
