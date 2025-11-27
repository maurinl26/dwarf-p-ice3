# -*- coding: utf-8 -*-
from contextlib import contextmanager
from typing import List, Tuple

from gt4py.storage import zeros
from gt4py.cartesian.gtscript import IJ, IJK, I, J, K
from ..utils.env import DTYPES, BACKEND


@contextmanager
def managed_temporaries(
    temporaries: List[Tuple[Tuple[int, ...], str]],
    domain: Tuple[int, 3],
    backend: str = BACKEND,
    dtypes: dict = DTYPES,
    aligned_index: Tuple[int, ...] = (0, 0, 0),
):

    def _allocate_temporary(dims, dtype):

        match dims:
            case _ if dims == IJ:
                return zeros(
                    shape=domain[:2],
                    dtype=dtypes[dtype],
                    aligned_index=aligned_index,
                    backend=backend,
                )
            case _ if dims == IJK:
                return zeros(
                    shape=domain,
                    dtype=dtypes[dtype],
                    aligned_index=aligned_index,
                    backend=backend,
                )

    yield from (_allocate_temporary(domain, dtype) for domain, dtype in temporaries)
