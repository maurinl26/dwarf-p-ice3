from __future__ import annotations

from gt4py.cartesian.gtscript import Field, function

from ifs_physics_common.framework.stencil import stencil_collection

@stencil_collection("ice4_rimltc")
def ice4_rimltc(
    t: Field["float"],
    exn: Field["float"],
    lv_fact: Field["float"],
    ls_fact: Field["float"],
    tht: Field["float"],            # theta at time t
    ri_in: Field["float"],          # rain water mixing ratio at t
    rimltc_mr_out: Field["float"],
    ld_compute: Field["float"]      # mask of computation
):
    
    from __externals__ import (
        tt,
        lfeedbackt,
    )
    
    if ri_in > 0 and t > tt and ld_compute == 1:
        rimltc_mr_out = ri_in
        if lfeedbackt:
            rimltc_mr_out = min(rimltc_mr_out, max(0, (tht -tt / exn ) / (ls_fact - lv_fact)))
            
    else :
        rimltc_mr_out= 0
    
  
            
    # TODO : stencil after rrhong in tendencies (3.3)