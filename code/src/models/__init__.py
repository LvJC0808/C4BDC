__all__ = [
    "LGBModel",
    "DoubleEnsembleModel",
]


def __getattr__(name: str):
    if name in {"LGBModel", "DoubleEnsembleModel"}:
        from .lgb_de import DoubleEnsembleModel, LGBModel

        return {"LGBModel": LGBModel, "DoubleEnsembleModel": DoubleEnsembleModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
