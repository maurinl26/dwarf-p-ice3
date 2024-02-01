from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rg.F90")
@stencil_collection("ice4_fast_rg")
def ice4_fast_rg(
    lcompute: Field["int"],     # mask for computation
    t: Field["float"],          # temperature
    rhodref: Field["float"],
    ri_in: Field["float"],      # Pristine ice mixing ratio at time t (PRIT in f90)
    rg_in: Field["float"],      # Graupel mixing ratio at time t
    rc_in: Field["float"],      # Cloud droplets mixing ratio at time t
    rs_in: Field["float"],      # Snow mixing ratio at time t
    ci_t: Field["float"],       # Pristine ice concentration at time t
    lbdar: Field["float"],      # Slope parameter of the raindrop distribution
    lbdas: Field["float"],      # Slope parameter of the aggregate distribution
    lbdag: Field["float"],      # Slope parameter of the graupel distribution
    ricfrrg: Field["float"],    # Rain contact freezing
    rrcfrig: Field["float"],    # Rain contact freezing
    ricfrr: Field["float"],     # Rain contact freezing
    ldsoft: "int",
    rg_rcdry_tnd: Field["float"],      # individual tendencies PRG_TEND in f90
    rg_ridry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
    rg_rsdry_tnd: Field["float"],
    rg_rswet_tnd: Field["float"],
    gdry: Field["int"]
):
    
    from __externals__  import (
        Ci, Cl, tt,
        lvtt,
        i_rtmin,
        r_rtmin,
        g_rtmin,
        s_rtmin,
        icfrr,
        rcfri,
        exicfrr,
        exrcfri, 
        cexvt,
        crflimit, # True to limit rain contact freezing to possible heat exchange
        cxg,
        dg,
        fcdryg,
        fidryg,
        colexig,
        colig
    )
    
    # 6.1 rain contact freezing
    with computation(PARALLEL), interval(...):
        
        if ri_in > i_rtmin and rr_in > r_rtmin and lcompute[0, 0, 0]:
            
            if ldsoft == 0:
        
                ricfrrg = icfrr * ri_in * lbdar ** exicfrr * rhodref ** (-cexvt)
                rrcfrig = rcfri * ci_t * lbdar ** exrcfri * rhodref ** (-cexvt)
        
                if crflimit:
                    zw0d = max(0, min(1, (ricfrrg * Ci + rrcfrig * Cl) * (tt - t) / max(1e-20, lvtt * rrcfrig)))
                    rrcfrig = zw0d * rrcfrig
                    ricffr = (1 - zw0d) * rrcfrig
                    ricfrrg = zw0d * ricfrrg
                
                else:
                    ricfrr = 0
                
        else:
            ricfrrg = 0
            rrcfrig = 0
            ricfrr = 0
    
    # 6.3 compute graupel growth
    with computation(PARALLEL), interval(...):
        
        if rg_in > g_rtmin and rc_in > r_rtmin and lcompute:
            if ldsoft == 0: # Not LDSOFT
                
                rg_rcdry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
                rg_rcdry_tnd = rg_rcdry_tnd * fcdryg * rc_in
                
        else:
            rg_rcdry_tnd = 0
            
        if rg_in > g_rtmin and ri_in > i_rtmin and lcompute:
            rg_ridry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
            rg_ridry_tnd = fidryg * exp(colexig * (t - tt)) * ri_in * rg_ridry_tnd
            rg_riwet_tnd = rg_ridry_tnd / (colig * exp(colexig * (t - tt)))
            
        else:
            rg_ridry_tnd = 0
            rg_riwet_tnd = 0
            
    # wet and dry collection of rs on graupel
    # l171 on f90
    with computation(PARALLEL), interval(...):
        
        if rs_in > s_rtmin and rg_in > g_rtmin and lcompute:
            gdry = 1 # GDRY is a boolean field in f90
            
        else:
            gdry = 0
            rg_rsdry_tnd = 0
            rg_rswet_tnd = 0
    
    # TODO : choose #ifdef flag REPRO48
    # with computation(PARALLEL), interval(...):
        
                
                