"""Dataset generation, on-disk layout and importers for real annotations."""

from .synth import CLASSES, Annotation, SynthConfig, config_for_split, generate_sequence

__all__ = [
    "CLASSES",
    "Annotation",
    "SynthConfig",
    "config_for_split",
    "generate_sequence",
]
