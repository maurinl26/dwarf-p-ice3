# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, exp, log

from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(from_file="PHYEX/src/common/micro/ice4_nucleation.func.h")
@stencil_collection("ice4_nucleation")
def ice4_fast_rg(
   ldcompute: Field["int"], 
   tht: Field["float"],
   pabs_t: Field["float"],
   rhodref: Field["float"],
   exn: Field["float"],
   ls_fact: Field["float"],
   t: Field["float"],
   rv_t: Field["float"],
   ci_t: Field["float"],
   rv_heni_mr: Field["float"],
   usw: Field["float"],     # PBUF(USW) in Fortran
   w2: Field["float"],
   w1: Field["float"],
   ssi: Field["float"]
):
    """Compute nucleation

    Args:
        ldcompute (Field[int]): _description_
        tht (Field[float]): _description_
        pabs_t (Field[float]): _description_
        rhodref (Field[float]): _description_
        exn (Field[float]): _description_
        ls_fact (Field[float]): _description_
        t (Field[float]): _description_
        rv_t (Field[float]): _description_
        ci_t (Field[float]): _description_
        rv_heni_mr (Field[float]): _description_
    """
    
    from __externals__ import (
        tt,
        v_rtmin,
        alpi,
        betai,
        gami,
        alpw,
        betaw,
        gamw,
        epsilo,
        nu20,
        alpha2,
        beta2,
        nu10,
        beta1,
        alpha1, 
        mnu0,
        lfeedbackt
    )
    
    # l66
    # TODO : use the assumptions directly
    with computation(PARALLEL), interval(...):
        if ldcompute == 1:
            # buf to be renamed
            lbuf_tmp = t < tt and rv_t > v_rtmin 
        else:
            lbuf_tmp = 0
            
    #l72
    with computation(PARALLEL), interval(...):
        
        if t < tt and rv_t > v_rtmin and ldcompute == 1:
            usw = 0
            w2 = 0
        
            w2 = log(t)
            usw = exp(alpw - betaw / t - gamw * w2)
            w2 = exp(alpi - betai / t - gami * w2)   
    
    # l83
    with computation(PARALLEL), interval(...):
    
        if t < tt and rv_t > v_rtmin and ldcompute == 1:
            
            ssi = 0
            w2 = min(pabs_t / 2, w2)
            ssi = rv_t * (pabs_t - w2) / (epsilo * w2) - 1
            # supersaturation over ice
        
            usw = min(pabs_t / 2, usw)
            usw = (usw / w2) * ((pabs_t - w2) / (pabs_t - usw))
            # supersaturation of saturated water vapor over ice
        
            ssi = min(ssi, usw) # limitation of ssi according to ssw = 0
        
    # l96
    with computation(PARALLEL), interval(...):
        
        w2 = 0
        if t < tt and rv_t > v_rtmin and ldcompute == 1:       
            if t < tt - 5 and ssi > 0:
                w2 = nu20 * exp(alpha2 * ssi - beta2)
            elif t < tt -2 and t > tt -5 and ssi > 0:
                w2 = max(nu20 * exp(-beta2), nu10 * exp(-beta1 * (t - tt)) * (ssi / usw) ** alpha1)
            
    
    # l107
    with computation(PARALLEL), interval(...):
        w2 = w2 - ci_t
        w2 = min(w2, 5e4)
        
    # l114 
    with computation(PARALLEL), interval(...):
        rv_heni_mr = 0
        if t < tt and rv_t > v_rtmin and ldcompute == 1:  
            rv_heni_mr = max(w2, 0) * mnu0 / rhodref   
            rv_heni_mr = min(rv_t, rv_heni_mr)
            
    # l122 
    with computation(PARALLEL), interval(...):
        if lfeedbackt:
            w1 = 0
            if t < tt and rv_t > v_rtmin and ldcompute == 1: 
                w1 = min(rv_heni_mr, max(0, (tt / exn - tht)) / ls_fact) / max(rv_heni_mr, 1e-20)
                
            rv_heni_mr *= w1
            w2 *= w1
            
    # l134
    with computation(PARALLEL), interval(...):
        if t < tt and rv_t > v_rtmin and ldcompute == 1: 
            ci_t = max(w2 + ci_t, ci_t)
             

        