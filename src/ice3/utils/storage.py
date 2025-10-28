from contextlib import contextmanager
from typing import List, Tuple

from gt4py.storage import zeros
from ..utils.env import DTYPES, BACKEND


@contextmanager
def managed_temporaries(
    temporaries: List[Tuple[Tuple[int, ...], str]],
    backend: str = BACKEND,
    aligned_index: Tuple[int, ...] = (0, 0, 0),
):
    def _allocate_temporary(domain, dtype):
        return zeros(
            shape=domain,
            dtype=DTYPES[dtype],
            aligned_index=aligned_index,
            backend=backend,
        )

    yield from (_allocate_temporary(domain, dtype) for domain, dtype in temporaries)
