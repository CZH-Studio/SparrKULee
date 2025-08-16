from typing import Optional

from .basemodel import MatchMismatchModel

from .baseline import BaselineModel
from .baseline_shared import SharedBaselineModel
# from .BMMSNet.dense_net import BMMSNet
from .sota import SOTAModel, SOTAParallelModel

from .clip import CLIPPretrainedModel, CLIPClsModel


def get_model(cfg) -> Optional[MatchMismatchModel]:
    model_name = cfg.model.model_name
    settings = {
        "baseline": {
            "model": BaselineModel,
            "kwargs": {
                "hidden_dim": cfg.model.baseline.hidden_dim
            },
        },
        "baseline_shared": {
            "model": SharedBaselineModel,
            "kwargs" :{
                "hidden_dim": cfg.model.baseline_shared.hidden_dim
            }
        },
        # "BMMSNet": {
        #     "model": BMMSNet,
        #     "kwargs": {}
        # },
        "sota": {
            "model": SOTAModel,
            "kwargs": {
                "hidden_dim": cfg.model.sota.hidden_dim,
            }
        },
        "sota_ensemble": {
            "model": None,
            "kwargs": {}
        },
        "sota_parallel": {
            "model": SOTAParallelModel,
            "kwargs": {
                "hidden_dim": cfg.model.sota_parallel.hidden_dim,
                "n_parallel": cfg.model.sota_parallel.n_parallel,
                "use_checkpoint": cfg.model.sota_parallel.use_checkpoint,
            }
        },
        "clip_pretrained": {
            "model": CLIPPretrainedModel,
            "kwargs": {
                "hidden_dim": cfg.model.clip.hidden_dim,
                "temperature": cfg.model.clip.temperature,
            }
        },
        "clip_cls": {
            "model": CLIPClsModel,
            "kwargs": {
                "hidden_dim": cfg.model.clip.hidden_dim,
            }
        },
        "clip_ensemble": {
            "model": None,
            "kwargs": {}
        },
    }
    if (setting := settings.get(model_name, None)) is None:
        raise KeyError(f"Model {model_name} not found in settings")
    if (model := setting.get("model", None)) is None:
        return None
    return model(**setting["kwargs"])
