"""SPS photometry emulators.

``load_emulator`` loads any checkpoint; ``emulator.spec`` (:class:`EmulatorSpec`)
describes its inputs; :class:`EmulatorView` presents it under another parameter
or band list.  Concrete classes live in the submodules.
"""

from arachne.emulator.base import SPSEmulator
from arachne.emulator.io import load_emulator, register_emulator_format
from arachne.emulator.spec import KNOWN_PARAMS, EmulatorSpec, ParamInfo, param_info
from arachne.emulator.view import EmulatorView

__all__ = [
    "KNOWN_PARAMS",
    "EmulatorSpec",
    "EmulatorView",
    "ParamInfo",
    "SPSEmulator",
    "load_emulator",
    "param_info",
    "register_emulator_format",
]
