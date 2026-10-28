from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax


@dataclass
class DomainData:
    """Pre-processed domain data for training.

    Attributes:
        context: the point and tensor arrays by tag.
        dimension: the spatial dimension.
    """

    context: Dict[str, jax.Array]
    dimension: int
