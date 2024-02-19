# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function
from gt4py.cartesian.gtscript import exp, log, sqrt, floor, atan
from ice3_gt4py.functions.compute_ice_frac import compute_frac_ice
from ice3_gt4py.functions.src_1d import src_1d
from ice3_gt4py.functions.temperature import update_temperature


from ifs_physics_common.framework.stencil import stencil_collection

#
@stencil_collection("upstream_sedimentation")
def upstream_sedimentation():

    NotImplemented
