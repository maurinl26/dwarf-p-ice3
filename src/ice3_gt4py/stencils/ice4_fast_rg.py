# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_fast_rg.F90")
@stencil_collection("ice4_fast_rg")
def ice4_fast_rg(
    ldcompute: Field["int"],
    t: Field["float"],
    rhodref: Field["float"],
    rv_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rg_t: Field["float"],
    rc_t: Field["float"],
    rs_t: Field["float"],
    ci_t: Field["float"],
    ka: Field["float"],
    dv: Field["float"],
    cj: Field["float"],
    lbdar: Field["float"],
    lbdas: Field["float"],
    lbdag: Field["float"],
    ricfrrg: Field["float"],
    rrcfrig: Field["float"],
    ricfrr: Field["float"],
    rg_rcdry_tnd: Field["float"],
    rg_ridry_tnd: Field["float"],
    rg_rsdry_tnd: Field["float"],
    rg_rrdry_tnd: Field["float"],
    rg_riwet_tnd: Field["float"],
    rg_rswet_tnd: Field["float"],
    rg_freez1_tnd: Field["float"],
    rg_freez2_tnd: Field["float"],
    rg_mltr: Field["float"],
    gdry: Field["int"],
    ldwetg: Field["int"], # bool, true if graupel grows in wet mode (out)
    lldryg: Field["int"], # linked to gdry + temporary
    rdryg_init_tmp: Field["float"],
    rwetg_init_tmp: Field["float"]
):
    """Compute fast graupel sources

    Args:
        ldcompute (Field[int]): switch to compute microphysical processes on column
        t (Field[float]): temperature
        rhodref (Field[float]): reference density
        ri_t (Field[float]): ice mixing ratio at t
        
        rg_t (Field[float]): graupel m.r. at t
        rc_t (Field[float]): cloud droplets m.r. at t
        rs_t (Field[float]): snow m.r. at t
        ci_t (Field[float]): _description_
        dv (Field[float]): diffusivity of water vapor
        ka (Field[float]): thermal conductivity of the air
        cj (Field[float]): function to compute the ventilation coefficient 
        lbdar (Field[float]): slope parameter for rain
        lbdas (Field[float]): slope parameter for snow
        lbdag (Field[float]): slope parameter for graupel
        ricfrrg (Field[float]): rain contact freezing
        rrcfrig (Field[float]): rain contact freezing
        ricfrr (Field[float]): rain contact freezing
        rg_rcdry_tnd (Field[float]): Graupel wet growth
        rg_ridry_tnd (Field[float]): Graupel wet growth
        rg_riwet_tnd (Field[float]): Graupel wet growth
        rg_rsdry_tnd (Field[float]): Graupel wet growth
        rg_rswet_tnd (Field[float]): Graupel wet growth
        gdry (Field[int]): _description_
    """

    from __externals__ import (
        Ci,
        Cl,
        tt,
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
        crflimit,  # True to limit rain contact freezing to possible heat exchange
        cxg,
        dg,
        fcdryg,
        fidryg,
        colexig,
        colig,
        ldsoft,
        estt,
        Rv,
        cpv,
        lmtt,
        o0depg,
        o1depg,
        ex0depg,
        ex1depg,
        levlimit,
        alpi,
        betai,
        gami
    )

    # 6.1 rain contact freezing
    with computation(PARALLEL), interval(...):

        if ri_t > i_rtmin and rr_t > r_rtmin and ldcompute == 1:
            
            # not LDSOFT : compute the tendencies
            if ldsoft == 0:

                ricfrrg = icfrr * ri_t * lbdar**exicfrr * rhodref ** (-cexvt)
                rrcfrig = rcfri * ci_t * lbdar**exrcfri * rhodref ** (-cexvt)

                if crflimit:
                    zw0d = max(
                        0,
                        min(
                            1,
                            (ricfrrg * Ci + rrcfrig * Cl)
                            * (tt - t)
                            / max(1e-20, lvtt * rrcfrig),
                        ),
                    )
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

        if rg_t > g_rtmin and rc_t > r_rtmin and ldcompute == 1:
            
            if ldsoft == 0:
                rg_rcdry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
                rg_rcdry_tnd = rg_rcdry_tnd * fcdryg * rc_t

        else:
            rg_rcdry_tnd = 0

        if rg_t > g_rtmin and ri_t > i_rtmin and ldcompute == 1:
            
            if ldsoft == 0:
                rg_ridry_tnd = lbdag ** (cxg - dg - 2.0) * rhodref ** (-cexvt)
                rg_ridry_tnd = fidryg * exp(colexig * (t - tt)) * ri_t * rg_ridry_tnd
                rg_riwet_tnd = rg_ridry_tnd / (colig * exp(colexig * (t - tt)))

        else:
            rg_ridry_tnd = 0
            rg_riwet_tnd = 0

    # 6.2.1 wet and dry collection of rs on graupel
    # Translation note : l171 in mode_ice4_fast_rg.F90
    with computation(PARALLEL), interval(...):

        if rs_t > s_rtmin and rg_t > g_rtmin and ldcompute == 1:
            gdry = 1  # GDRY is a boolean field in f90

        else:
            gdry = 0
            rg_rsdry_tnd = 0
            rg_rswet_tnd = 0
       
    # TODO: l181 to 243     

    # Translation note l300 to l316 removed (no hail)
    
    # Freezing rate and growth mode
    # l251
    with computation(PARALLEL), interval(...):
        
        if rg_t > g_rtmin and ldcompute == 1:
            
            # Duplicated code with ice4_fast_rs
            if ldsoft == 0:
                rg_freez1_tnd = rv_t * pres / (epsilo + rv_t)
                if levlimit:
                    rg_freez1_tnd = min(rg_freez1_tnd, exp(alpi - betai / t - gami * log(t)) )

                rg_freez1_tnd = ka * (tt - t) + dv * (lvtt + (cpv - Cl ) * (t - tt)) * (estt - rg_freez1_tnd) / (Rv * t)
                rg_freez1_tnd *= (o0depg * lbdag ** ex0depg + o1depg * cj * lbdag ** ex1depg) / (rhodref * (lmtt - Cl * (tt - t)))
                rg_freez2_tnd = (rhodref * (lmtt + (Ci - Cl) * (tt - t))) / (rhodref * (lmtt - Cl * (tt - t)))
         
            rwetg_init_tmp = max(rg_riwet_tnd + rg_rswet_tnd, max(0, rg_freez1_tnd + rg_freez2_tnd * (rg_riwet_tnd + rg_rswet_tnd)))
            
            # Growth mode
            # TODO : convert logical to int operations
            ldwetg = max(0, rwetg_init_tmp - rg_riwet_tnd - rg_rswet_tnd) <= max(rdryg_init_tmp - rg_ridry_tnd - rg_rsdry_tnd)
                 
            if lnullwetg == 1:
                ldwetg = ldwetg and rdryg_init_tmp > 0
            else:
                ldwetg = ldwetg and rwetg_init_tmp >0
                
            if lwetgpost == 0:
                ldwetg = ldwetg and t < tt
            
            lldryg = t < tt and rdryg_init_tmp and max(0, rwetg_init_tmp - rg_riwet_tnd - rg_rswet_tnd) > max(0, rg_rsdry_tnd - rg_ridry_tnd - rg_rsdry_tnd)
            
        else:
            rg_freez1_tnd = 0
            rg_freez2_tnd = 0
            rwetg_init_tmp = 0
            ldwetg == 0
            lldryg == 0
        
    # l317 
    with computation(PARALLEL), interval(...):
        
        if ldwetg == 1:
            # TODO : rwetg_init_tmp to instanciate
            rr_wetg = -(rg_riwet_tnd + rg_rswet_tnd + rg_rcdry_tnd - rwetg_init_tmp)
            rc_wetg = rg_rcdry_tnd
            ri_wetg = rg_riwet_tnd
            rs_wetg = rg_rswet_tnd
            
        else:
            rr_wetg = 0
            rc_wetg = 0
            ri_wetg = 0
            rs_wetg = 0
            
        if lldryg == 1:
            rc_dry = rg_rcdry_tnd
            rr_dry = rg_rrdry_tnd
            ri_dry = rg_ridry_tnd
            rs_dry = rg_rsdry_tnd
            
        else:
            rc_dry = 0
            rr_dry = 0
            ri_dry = 0
            rs_dry = 0
            
            
    
    # 6.5 Melting of the graupel
    with computation(PARALLEL), interval(...):
        
        if rg_t > g_rtmin and t > tt and ldcompute == 1:
            if ldsoft == 0:
                rg_mltr = rv_t * pres / (epsilo + rv_t)
                if levlimit == 1:
                    rg_mltr = min(rg_mltr, exp(alpw - betaw / t - gamw * log(t)))
                    
                rg_mltr = ka * (tt - t) + dv * (lvtt + (cpv - Cl ) * (t - tt)) * (estt - rg_mltr) / (Rv * t) 
                rg_mltr = max(0, (-rg_mltr * (o0depg * lbdag ** ex0depg + o1depg * cj * lbdag ** ex1depg) - (rg_rcdry_tnd + rg_rrdry_tnd) * (rhodref * Cl * (tt - t))) / (rhodref * lmtt))

        else:
            rg_mltr = 0
            
