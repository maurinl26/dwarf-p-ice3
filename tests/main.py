# -*- coding: utf-8 -*-
import typer

# -*- coding: utf-8 -*-
import itertools
from ifs_physics_common.framework.config import GT4PyConfig
import sys
import logging
import typer

from stencils.test_compile_stencils import STENCIL_COLLECTIONS, build
from ice3_gt4py.phyex_common.phyex import Phyex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = typer.Typer()


@app.command()
def test_compile_stencils(backend: str):
    """Compile the list of stencils with given backend"""

    # Compiling with phyex externals
    for backend, stencil_collection in itertools.product(
        [backend], STENCIL_COLLECTIONS
    ):

        logging.info("Building with Phyex externals")
        config = GT4PyConfig(
            backend=backend, rebuild=True, validate_args=True, verbose=True
        )
        phyex = Phyex("AROME")
        build(phyex.to_externals(), backend, config, stencil_collection)


if __name__ == "__main__":
    app()
