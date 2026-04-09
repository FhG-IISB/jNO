from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax


@dataclass
class DomainData:
    """Pre-processed domain data for training."""

    context: Dict[str, jax.Array]
    dimension: int
