from .basemodel import MatchMismatchModel

from .baseline import BaselineModel
from .baseline_shared import SharedBaselineModel
from .BMMSNet.dense_net import BMMSNet
from .sota import SOTAModel
from .clip import CLIPPretrainedModel, CLIPClsModel


def get_model(cfg) -> MatchMismatchModel:
    model_name = cfg.model.model_name
    m = {
        "baseline": BaselineModel,
        "baseline_shared": SharedBaselineModel,
        "BMMSNet": BMMSNet,
        "sota": SOTAModel,
        "clip_pretrained": CLIPPretrainedModel,
        "clip_cls": CLIPClsModel,
    }
    p = {
        "baseline": {
            "hidden_dim": cfg.model.baseline.hidden_dim
        },
        "baseline_shared": {
            "hidden_dim": cfg.model.baseline_shared.hidden_dim
        },
        "BMMSNet": {},
        "sota": {
            "hidden_dim": cfg.model.sota.hidden_dim,
        },
        "clip_pretrained": {
            "hidden_dim": cfg.model.clip.hidden_dim,
            "temperature": cfg.model.clip.temperature,
        },
        "clip_cls": {
            "hidden_dim": cfg.model.clip.hidden_dim,
        },
    }
    return m[model_name](**p[model_name])


