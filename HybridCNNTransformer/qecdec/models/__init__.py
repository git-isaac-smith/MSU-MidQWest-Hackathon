from ..config import Config
from .cnn_baseline import CNNBaseline
from .hybrid import HybridDecoder


def build_model(cfg: Config, max_d: int = 9):
    """cfg.model in {"hybrid", "cnn_baseline"}; max_d is only used for the CNN receptive-field check."""
    if cfg.model == "hybrid":
        return HybridDecoder(cfg)
    if cfg.model == "cnn_baseline":
        return CNNBaseline(cfg, max_d=max_d)
    raise ValueError(f"unknown model {cfg.model!r}")
