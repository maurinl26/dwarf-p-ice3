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
    c_sed_tmp: Field["float"],      # sedimentation source for cloud droplets ZSED
    r_sed_tmp: Field["float"],      # sedimentation source for rain 
    g_sed_tmp: Field["float"],      # sedimentation source for graupel
    i_sed_tmp: Field["float"],      # sedimentation source for ice
    s_sed_tmp: Field["float"],      # sedimentation source for snow    
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
        c_rtmin,         # cloud droplet rc min  
        r_rtmin,
        lbexc,
        cc,
        dc,
        cexvt,
        fsedc,
        fsedr, 
        exsedr,
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
            
            qp_tmp = c_sed_tmp[0, 0, 1] * dt__rho_dz_tmp[0, 0, 0]
            if rc_in > c_rtmin or qp_tmp > c_rtmin:
                
                if rc_in > c_rtmin:
                    wsedw1_tmp = terminal_velocity(
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


                if qp_tmp > c_rtmin:
                    wsedw2_tmp = terminal_velocity(
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
        # end lsedic      
        # END SUBROUTINE
            
        # 2.2 rain
        # Other species            
        qp_tmp[0, 0, 0] = r_sed_tmp[0, 0, 1] * dt__rho_dz_tmp[0, 0, 0]
        if rr_in > r_rtmin or qp_tmp > r_rtmin:
            if rr_in > r_rtmin:
                wsedw1_tmp = other_species(
                        fsedr, 
                        exsedr,
                        rr_in,
                        rhodref
                    )
            else:
                wsedw1_tmp = 0
                    
            if qp_tmp > r_rtmin:
                wsedw2_tmp = other_species(
                        fsedr,
                        exsedr,
                        qp_tmp,
                        rhodref
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
                
        # 2.3 ice
        
        
        # 2.4 snow
        qp_tmp[0, 0, 0] = r_sed_tmp[0, 0, 1] * dt__rho_dz_tmp[0, 0, 0]
        if rr_in > r_rtmin or qp_tmp > r_rtmin:
            if rr_in > r_rtmin:
                wsedw1_tmp = other_species(
                        fsedr, 
                        exsedr,
                        rr_in,
                        rhodref
                    )
            else:
                wsedw1_tmp = 0
                    
            if qp_tmp > r_rtmin:
                wsedw2_tmp = other_species(
                        fsedr,
                        exsedr,
                        qp_tmp,
                        rhodref
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
        
        # 2.5 graupel
        qp_tmp[0, 0, 0] = r_sed_tmp[0, 0, 1] * dt__rho_dz_tmp[0, 0, 0]
        if rr_in > r_rtmin or qp_tmp > r_rtmin:
            if rr_in > r_rtmin:
                wsedw1_tmp = other_species(
                        fsedr, 
                        exsedr,
                        rr_in,
                        rhodref
                    )
            else:
                wsedw1_tmp = 0
                    
            if qp_tmp > r_rtmin:
                wsedw2_tmp = other_species(
                        fsedr,
                        exsedr,
                        qp_tmp,
                        rhodref
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
                    
        
    # 3. Sources 
    # Calcul des tendances
    with computation(PARALLEL), interval(...):
        
        rcs_tnd = rcs_tnd + dt__rho_dz_tmp * (c_sed_tmp[0, 0, 1] - c_sed_tmp[0, 0, 0]) / dt
        ris_tnd = ris_tnd + dt__rho_dz_tmp * (i_sed_tmp[0, 0, 1] - i_sed_tmp[0, 0, 0]) / dt
        rss_tnd = rss_tnd + dt__rho_dz_tmp * (s_sed_tmp[0, 0, 1] - s_sed_tmp[0, 0, 0]) / dt
        rgs_tnd = rgs_tnd + dt__rho_dz_tmp * (g_sed_tmp[0, 0, 1] - g_sed_tmp[0, 0, 0]) / dt
        rrs_tnd = rrs_tnd + dt__rho_dz_tmp * (r_sed_tmp[0, 0, 1] - r_sed_tmp[0, 0, 0]) / dt
    
    # Instantaneous fluxes
    with computation(PARALLEL), interval(0, 1):    
        
        inst_rc_out = c_sed_tmp / rholw
        inst_rr_out = r_sed_tmp / rholw
        inst_ri_out = i_sed_tmp / rholw
        inst_rs_out = s_sed_tmp / rholw
        inst_rg_out = g_sed_tmp / rholw
         

@function
def terminal_velocity(
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
    wlbda_tmp = 6.6e-8 * (101325 / pabst[0, 0, 0]) * (tht[0, 0, 0] / 293.15)
    wlbdc_tmp = (lbc * conc3d / (rhodref * content)) ** lbexc
    cc_tmp = cc * (1 + 1.26 * wlbda_tmp * wlbdc_tmp / ray)
    wsedw1 = rhodref ** (-cexvt) * wlbdc_tmp * (-dc) * cc_tmp * fsedc
    
    return wsedw1

# FWSED1
# P1 in Bouteloup paper
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
#P2 in Bouteloup paper
@function
def weighted_sedimentation_flux_2(
    wsedw: Field["float"],
    wsedsup: Field["float"],
    dz: Field["float"],
    dt: "float"
):
    
    return max(0, 1 - dz / (dt * wsedw)) * wsedsup[0, 0, 1]


@function
def other_species(
    fsed: "float",
    exsed: "float",
    content_in: Field["float"],
    rhodref: Field["float"]
):
    
    from __externals__  import cexvt
    
    wsedw = fsed * content_in * (exsed - 1) * rhodref ** (exsed - cexvt - 1)
    
    return wsedw


     