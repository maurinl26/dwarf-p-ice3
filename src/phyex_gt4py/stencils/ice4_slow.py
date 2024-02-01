from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function

from ifs_physics_common.framework.stencil import stencil_collection

@stencil_collection("ice4_rimltc")
def ice4_slow(
    rhodref: Field["float"],
    t: Field["float"],
    ssi: Field["float"],    # supersaturation over ice
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    rv_in: Field["float"], # vapour mr at time t
    rc_in: Field["float"], 
    ri_in: Field["float"],
    rs_in: Field["float"],
    rg_in: Field["float"],
    lbdas: Field["float"],  # Slope parameter of the aggregate distribution
    lbdag: Field["float"],  # Slope parameter of the graupel distribution
    ai: Field["float"],     # Thermodynamical function PAI
    cj: Field["float"],     # to compute the ventilation coeff
    hli_hcf: Field["float"],
    hli_hri: Field["float"],
    rc_honi_tnd: Field["float"],
    rv_deps_tnd: Field["float"],
    ri_aggs_tnd: Field["float"],
    ri_auts_tnd: Field["float"],
    rv_depg_tnd: Field["float"]
):
    
    from __externals__ import (
        
    )
    
    
    
  
            
    # TODO : stencil after rrhong in tendencies (3.3)