"""Gloas consensus spec implementation.

Note: The 'types' module must be imported after calling constants.set_preset()
to ensure SSZ types have correct sizes for the chosen preset.
"""

from . import constants
from .network_config import NetworkConfig, get_config, load_config

__all__ = ["constants", "NetworkConfig", "get_config", "load_config"]
