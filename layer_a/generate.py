"""Compatibility import for model.layer_a.generate."""
from model.layer_a.generate import *  # noqa: F401,F403
from model.layer_a import generate as _implementation

for _name in dir(_implementation):
    if _name.startswith("_"):
        globals()[_name] = getattr(_implementation, _name)
