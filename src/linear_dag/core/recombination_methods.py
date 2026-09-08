"""Shared selection for the recombination implementations."""

from .recombination import Recombination


def _recombination_class(method):
    if method == "legacy":
        return Recombination
    if method == "msbf":
        from .recombination_msbf import Recombination as MSBFRecombination

        return MSBFRecombination
    raise ValueError(f"Unknown recombination method {method!r}; choose 'legacy' or 'msbf'")
