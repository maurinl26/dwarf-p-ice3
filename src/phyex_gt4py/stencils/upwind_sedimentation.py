# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function
from gt4py.cartesian.gtscript import exp, log, sqrt, floor, atan
from phyex_gt4py.functions.compute_ice_frac import compute_frac_ice
from phyex_gt4py.functions.src_1d import src_1d
from phyex_gt4py.functions.temperature import update_temperature


from phyex_gt4py.functions.ice_adjust import (
    vaporisation_latent_heat,
    sublimation_latent_heat,
)
from ifs_physics_common.framework.stencil import stencil_collection

# 
@stencil_collection("upstream_sedimentation")
def upstream_sedimentation():
    
    NotImplemented