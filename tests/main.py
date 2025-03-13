# -*- coding: utf-8 -*-
import itertools
from ifs_physics_common.framework.config import GT4PyConfig
import sys
import logging
import typer

from stencils.test_compile_stencils import STENCIL_COLLECTIONS, build
from ice3_gt4py.phyex_common.phyex import Phyex
from components.test_component import build_component

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


@app.command()
def test_components(backend: str):
    """Test ImplicitTendencyComponents for parts of model"""

    from ice3_gt4py.components.aro_adjust import AroAdjust

    logging.info(f"Testing AroAdjust on backend {backend}")
    build_component(backend, AroAdjust)

    from ice3_gt4py.components.aro_filter import AroFilter

    logging.info(f"Testing AroFilter on backend {backend}")
    build_component(backend, AroFilter)

    from ice3_gt4py.components.ice_adjust import IceAdjust

    logging.info(f"Testing IceAdjust on backend {backend}")
    build_component(backend, IceAdjust)

    from ice3_gt4py.components.ice4_tendencies import Ice4Tendencies

    logging.info(f"Testing Ice4Tendencies on backend {backend}")
    build_component(backend, Ice4Tendencies)


if __name__ == "__main__":
    app()
