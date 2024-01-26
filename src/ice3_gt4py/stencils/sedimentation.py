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


@stencil_collection("statistical_sedimentation")
def sedimentation_stat(
    dt: "float",
    rhodref: Field["float"],        # reference density
    dz: Field["float"],
    pabst: Field["float"],          # absolute pressure at t
    tht: Field["float"],            # potential temperature at t
    rc_in: Field["float"],          # droplet content at t 
    rr_in: Field["float"],          # rain content at t
    ri_in: Field["float"],          # ice content at t 
    rs_in: Field["float"],          # snow content at t 
    rg_in: Field["float"],          # graupel content at t
    rcs_tnd: Field["float"],        # droplet content tendency PRCS
    rrs_tnd: Field["float"],        # rain content tendency PRRS
    ris_tnd: Field["float"],        # ice content tendency PRIS
    rss_tnd: Field["float"],        # snow content tendency PRSS
    rgs_tnd: Field["float"],        # graupel conntent tendency PRGS
    dt__rho_dz_tmp: Field["float"], # ZTSORHODZ delta t over rho x delta z 
    sea_mask: Field["int"],         # Mask for sea PSEA
    town_fraction: Field["float"],  # Fraction of map which is town PTOWN
    wgt_lbc_tmp: Field["float"],    # LBC weighted by sea fraction
    sed_tmp: Field["float"],        # sedimentation source ZSED 
    # cloud subroutine
    qp_tmp: Field["float"],
    wlbda_tmp: Field["float"],      # 
    wlbdc_tmp: Field["float"],      # 
    cc_tmp: Field["float"],         # sedimentation fall speed
    wsedw1: Field["float"],         # 
    wsedw2: Field["float"],         #
    lbc_tmp: Field["float"],        # 
    ray_tmp: Field["float"],        # Cloud mean radius ZRAY
    conc3d_tmp: Field["float"],     # sea and urban modifications
    # diagnostics
    fpr_out: Field["float"],        # precipitation flux through upper face of the cell
    inst_rr_out: Field["float"],    # instant rain precipitation PINPRR
    inst_rc_out: Field["float"],    # instant droplet precipitation PINPRC
    inst_ri_out: Field["float"],    # instant ice precipitation PINPRI
    inst_rs_out: Field["float"],    # instant snow precipitation PINPRS
    inst_rg_out: Field["float"],    # instant graupel precipitation PINPRG
):
    
    from __externals__ import (
        lsedic, 
        rholw,          # volumic lass of liquid water  
        crtmin,         # cloud droplet rc min  
        lbexc,
        cc,
        dc,
        cexvt,
        fsedc           # Warning : as to be externalized
    )
    
    # Note Hail is omitted
    # Note : lsedic = True in Arome
    # Note : frp is sed_tmp
    # FRPR present for AROME config
    
    # 1. Compute the fluxes
    # Gamma computations shifted in RainIceDescr
    # Warning : call shift 
    
    # 2. Fluxes
    with computation(PARALLEL), interval(...):
        
        dt__rho_dz_tmp = dt / (rhodref * dz)
        
        # 2.1 cloud 
        if lsedic:
            # subroutine cloud in fortran
            # 1. ray, lbc, fsedc, conc3d
            
            qp_tmp = sed_tmp[0, 0, 1] * dt__rho_dz_tmp[0, 0, 0]
            if rc_in > crtmin or qp_tmp > crtmin:
                
                if rc_in > crtmin:
                    wsedw1_tmp = weighted_sedimentation_source(
                        rc_in,
                        tht,
                        pabst,
                        rhodref,
                        lbc_tmp,
                        ray_tmp,
                        conc3d_tmp 
                    )
                else:
                    wsedw1_tmp = 0


                if qp_tmp > crtmin:
                    wsedw2_tmp = weighted_sedimentation_source(
                        qp_tmp,
                        tht,
                        pabst,
                        rhodref,
                        lbc_tmp,
                        ray_tmp,
                        conc3d_tmp 
                    )
                else:
                    wsedw2_tmp = 0
            else:
                wsedw1_tmp = 0
                wsedw2_tmp = 0
                
            sed_tmp = weighted_sedimentation_flux_1(
                wsedw1_tmp,
                dz,
                rhodref,
                rc_in,
                dt
            )
            
            if wsedw2_tmp != 0:
                sed_tmp = sed_tmp + weighted_sedimentation_flux_2(
                    wsedw2_tmp,
                    dt,
                    dz,
                    sed_tmp
                )
                    
            # END SUBROUTINE
            
        # 2.2 rain
            
        
    # 3. Sources 
    with computation(PARALLEL), interval(...):
        
    NotImplemented
    
    
@function
def weighted_sedimentation_source(
    content: Field["float"],
    tht: Field["float"],
    pabst: Field["float"],
    rhodref: Field["float"],
    lbc: Field["float"],
    ray: Field["float"],
    conc3d: Field["float"]
):
    from __externals__ import (
        cc,
        lbexc,
        fsedc,
        cexvt,
        dc
    )
    wlbda_tmp = 6.6e-8 * (101325 / pabst[0, 0, 0])*(tht[0, 0, 0] / 293.15)
    wlbdc_tmp = (lbc * conc3d / (rhodref * content)) ** lbexc
    cc_tmp = cc * (1 + 1.26 * wlbda_tmp * wlbdc_tmp / ray)
    wsedw1 = rhodref ** (-cexvt) * wlbdc_tmp * (-dc) * cc_tmp * fsedc
    
    return wsedw1

# FWSED1
@function
def weighted_sedimentation_flux_1(
    wsedw: Field["float"],
    dz: Field["float"],
    rhodref: Field["float"],
    content_in: Field["float"],
    dt: "float"
):
    
    return min(rhodref * dz * content_in / dt, wsedw * rhodref, content_in)

#FWSED2
@function
def weighted_sedimentation_flux_2(
    wsedw: Field["float"],
    wsedsup: Field["float"],
    dz: Field["float"],
    dt: "float"
):
    
    return max(0, 1 - dz / (dt * wsedw)) * wsedsup[0, 0, 1]
