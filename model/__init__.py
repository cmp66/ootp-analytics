"""The model package. Importing it stays light: the conversions of export values are loaded at once, the legacy torch trainers
(`MLPexplicit`, `RegressionRunner`, `Modeler` of model/legacy.py) only when a name of them is first used, so that the viewer does
not load torch, shap and scikit-learn."""

from model.converters import *  # noqa: F401,F403
from model.converters import ratings_conversions_10, ratings_conversions_80  # noqa: F401

_LEGACY = ("MLPexplicit", "RegressionRunner", "Modeler")


def __getattr__(name):
    if name in _LEGACY:
        from model import legacy

        return getattr(legacy, name)
    raise AttributeError(f"module 'model' has no attribute {name!r}")
