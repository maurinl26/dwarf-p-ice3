# -*- coding: utf-8 -*-
from __future__ import annotations

from gt4py.cartesian.gtscript import Field, computation, interval, PARALLEL, IJ
from ifs_physics_common.framework.stencil import stencil_collection
from ifs_physics_common.utils.f2py import ported_method


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=693, to_line=728
)
@stencil_collection("rain_ice_total_tendencies")
def rain_ice_total_tendencies(
    wr_th: Field["float"],
    wr_v: Field["float"],
    wr_c: Field["float"],
    wr_r: Field["float"],
    wr_i: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    exnref: Field["float"],
    ths: Field["float"],
    rvs: Field["float"],
    rcs: Field["float"],
    rrs: Field["float"],
    ris: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
    rvheni: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
):
    """Update tendencies

    Args:
        wr_th (Field[float]): potential temperature initial value
        wr_v (Field[float]): vapour initial value
        wr_c (Field[float]): cloud droplets initial value
        wr_r (Field[float]): rain initial value
        wr_i (Field[float]): ice initial value
        wr_s (Field[float]): snow initial value
        wr_g (Field[float]): graupel initial value
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        lv_fact (Field[float]): vapourisation latent heat over heat capacity
        exnref (Field[float]): reference exner pressure
        ths (Field[float]): source (tendency) of potential temperature
        rvs (Field[float]): source (tendency) of vapour
        rcs (Field[float]): source (tendency) of cloud droplets
        rrs (Field[float]): source (tendency) of rain
        ris (Field[float]): source (tendency) of ice
        rss (Field[float]): source (tendency) of snow
        rgs (Field[float]): source (tendency) of graupel
        rvheni (Field[float]): _description_
        rv_t (Field[float]): vapour m.r. at t
        rc_t (Field[float]): droplets m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r. at t
        rg_t (Field[float]): graupel m.r. at t
    """

    from __externals__ import INV_TSTEP

    with computation(PARALLEL), interval(...):

        # Translation note ls, lv replaced by ls_fact, lv_fact

        # Hydrometeor tendency
        wr_v = (wr_v - rv_t) * INV_TSTEP
        wr_c = (wr_c - rc_t) * INV_TSTEP
        wr_r = (wr_r - rr_t) * INV_TSTEP
        wr_i = (wr_i - ri_t) * INV_TSTEP
        wr_s = (wr_s - rs_t) * INV_TSTEP
        wr_g = (wr_g - rg_t) * INV_TSTEP

        # Theta tendency
        wr_th = (wr_c + wr_r) * lv_fact + (wr_i + wr_s + wr_g) * ls_fact

        # Tendencies to sources, taking nucleation into account (rv_heni)
        ths += wr_th + rvheni * ls_fact
        rvs += wr_v - rvheni
        rcs += wr_c
        rrs += wr_r
        ris += wr_i + rvheni
        rss += wr_s
        rgs += wr_g


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=367, to_line=396
)
@stencil_collection("rain_ice_init")
def rain_ice_init(
    ldmicro: Field["bool"],
    exn: Field["float"],
    ls_fact: Field["float"],
    lv_fact: Field["float"],
    th_t: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
):
    """_summary_

    Args:
        ldmicro (Field[bool]): mask for microphysical computations
        exn (Field[float]): exner pressure
        ls_fact (Field[float]): sublimation latent heat over capacity
        lv_fact (Field[float]): vaporisation latent heat over capacity
        th_t (Field[float]): potential temperature at t
        rv_t (Field[float]): vapour m.r. at t
        rc_t (Field[float]): cloud droplet m.r. at t
        rr_t (Field[float]): rain m.r. at t
        ri_t (Field[float]): ice m.r. at t
        rs_t (Field[float]): snow m.r.
        rg_t (Field[float]): graupel m.r.
    """

    from __externals__ import (
        C_RTMIN,
        R_RTMIN,
        I_RTMIN,
        S_RTMIN,
        G_RTMIN,
        CPD,
        CPV,
        CI,
        CL,
        TT,
        LSTT,
        LVTT,
    )

    with computation(PARALLEL), interval(...):
        divider = CPD + CPV * rv_t + CL * (rc_t + rr_t) + CI * (ri_t + rs_t + rg_t)
        t = th_t * exn
        ls_fact = (LSTT + (CPV - CI) * (t - TT)) / divider
        lv_fact = (LVTT + (CPV - CL) * (t - TT)) / divider

        ldmicro = (
            rc_t > C_RTMIN
            or rr_t > R_RTMIN
            or ri_t > I_RTMIN
            or rs_t > S_RTMIN
            or rg_t > G_RTMIN
        )


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=424, to_line=444
)
@stencil_collection("initial_values_saving")
def initial_values_saving(
    wr_th: Field["float"],
    wr_v: Field["float"],
    wr_c: Field["float"],
    wr_r: Field["float"],
    wr_i: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    th_t: Field["float"],
    rv_t: Field["float"],
    rc_t: Field["float"],
    rr_t: Field["float"],
    ri_t: Field["float"],
    rs_t: Field["float"],
    rg_t: Field["float"],
    evap3d: Field["float"],
    rainfr: Field["float"],
):

    from __externals__ import LWARM

    with computation(PARALLEL), interval(...):
        wr_th = th_t
        wr_v = rv_t
        wr_c = rc_t
        wr_r = rr_t
        wr_i = ri_t
        wr_s = rs_t
        wr_g = rg_t

        # LWARM is True for AROME
        if __INLINED(LWARM):
            evap3d = 0
        rainfr = 0

@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=452, to_line=463
)
@stencil_collection("rain_ice_nucleation_pre_processing")
def rain_ice_nucleation_pre_processing(
    ldmicro: Field["bool"],
    ci_t: Field["float"],
    w3d: Field["float"],
    ls_fact: Field["float"],
    exn: Field["float"],
):
    """Preprocessing for nucleation step

    Args:
        ldmicro (Field[bool]): mask for microphysics computation
        ci_t (Field[float]): concentration of ice
        w3d (Field[float]): _description_
        ls_fact (Field[float]): sublimation latent heat over heat capacity
        exn (Field[float]): exner pressure
    """

    with computation(PARALLEL), interval(...):
        # Translation note : lw3d is (not ldmicro)
        # therefore, lw3d is removed from parameters
        if not ldmicro:
            w3d = ls_fact / exn
            ci_t = 0


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=473, to_line=477
)
@stencil_collection("rain_ice_nucleation_post_processing")
def rain_ice_nucleation_post_processing(
    rvs: Field["float"],
    rvheni: Field["float"],
):
    """rvheni limiter (heterogeneous nucleation of ice)

    Args:
        rvs (Field[float]): source of vapour
        rvheni (Field[float]): vapour mr change due to heni
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(...):
        rvheni = min(rvs, rvheni / TSTEP)


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=492, to_line=498
)
@stencil_collection("ice4_precipitation_fraction_sigma")
def ice4_precipitation_fraction_sigma(sigs: Field["float"], sigma_rc: Field["float"]):
    """Compute supersaturation variance with supersaturation standard deviation.

    In rain_ice.F90
    IF (PARAMI%CSUBG_AUCV_RC=='PDF ' .AND. PARAMI%CSUBG_PR_PDF=='SIGM')

    Args:
        sigs (Field[float]): _description_
        sigma_rc (Field[float]):
    """
    with computation(PARALLEL), interval(...):
        sigma_rc = sigs**2


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=492, to_line=498
)
@stencil_collection("ice4_precipitation_fraction_liquid_content")
def ice4_precipitation_fraction_liquid_content(
    hlc_lrc: Field["float"],
    hlc_hrc: Field["float"],
    hli_lri: Field["float"],
    hli_hri: Field["float"],
    hlc_lcf: Field["float"],
    hlc_hcf: Field["float"],
    hli_hcf: Field["float"],
    hli_lcf: Field["float"],
    rc_t: Field["float"],
    ri_t: Field["float"],
    cldfr: Field["float"],
):
    """Compute supersaturation variance with supersaturation standard deviation.

    In rain_ice.F90
    IF (PARAMI%CSUBG_AUCV_RC=='ADJU' .OR. PARAMI%CSUBG_AUCV_RI=='ADJU') THEN

    Args:
        hlc_lrc: Field["float"],
        hlc_hrc: Field["float"],
        hli_lri: Field["float"],
        hli_hri: Field["float"],
        hlc_lcf: Field["float"],
        hlc_hcf: Field["float"],
        hli_hcf: Field["float"],
        hli_lcf: Field["float"],
        rc_t: Field["float"],
        ri_t: Field["float"],
        cldfr: Field["float"]
    """
    with computation(PARALLEL), interval(...):
        hlc_lrc = rc_t - hlc_hrc
        hli_lri = ri_t - hli_hri
        hlc_lcf = cldfr - hlc_hcf if rc_t > 0 else 0
        hli_lcf = cldfr - hli_hcf if ri_t > 0 else 0


@ported_method(from_file="PHYEX/src/common/micro/mode_ice4_compute_pdf.F90")
@stencil_collection("ice4_compute_pdf")
def ice4_compute_pdf(
    ldmicro: Field["bool"],
    rhodref: Field["float"],
    rc_t: Field["float"],
    ri_t: Field["float"],
    cf: Field["float"],
    t: Field["float"],
    sigma_rc: Field["float"],
    hlc_hcf: Field["float"],
    hlc_lcf: Field["float"],
    hlc_hrc: Field["float"],
    hlc_lrc: Field["float"],
    hli_hcf: Field["float"],
    hli_lcf: Field["float"],
    hli_hri: Field["float"],
    hli_lri: Field["float"],
    rf: Field["float"],
):
    """PDF used to split clouds into high and low content parts

    Args:
        ldmicro (Field[bool]): mask for microphysics computation
        rc_t (Field[float]): cloud droplet m.r. estimate at t
        ri_t (Field[float]): ice m.r. estimate at t
        cf (Field[float]): cloud fraction
        t (Field[float]): temperature
        sigma_rc (Field[float]): standard dev of cloud droplets m.r. over the cell
        hlc_hcf (Field[float]): _description_
        hlc_lcf (Field[float]): _description_
        hlc_hrc (Field[float]): _description_
        hlc_lrc (Field[float]): _description_
        hli_hcf (Field[float]): _description_
        hli_lcf (Field[float]): _description_
        hli_hri (Field[float]): _description_
        hli_lri (Field[float]): _description_
        rf (Field[float]): _description_
    """

    from __externals__ import (
        CRIAUTC,
        C_RTMIN,
        SUBG_AUCV_RC,
        SUBG_PR_PDF,
        CRIAUTI,
        ACRIAUTI,
        BCRIAUTI,
        TT,
        SUBG_AUCV_RI,
        I_RTMIN,
    )

    with computation(PARALLEL), interval(...):
        rcrautc_tmp = CRIAUTC / rhodref if ldmicro else 0

    # HSUBG_AUCV_RC = NONE (0)
    with computation(PARALLEL), interval(...):

        # TODO: inline this choice
        if SUBG_AUCV_RC == 0:
            if rc_t > rcrautc_tmp and ldmicro:
                hlc_hcf = 1
                hlc_lcf = 0
                hlc_hrc = rc_t
                hlc_lrc = 0

            elif rc_t > C_RTMIN and ldmicro:
                hlc_hcf = 0
                hlc_lcf = 1
                hlc_hrc = 0
                hlc_lrc = rc_t

            else:
                hlc_hcf = 0
                hlc_lcf = 0
                hlc_hrc = 0
                hlc_lrc = 0

        # HSUBG_AUCV_RC = CLFR (1)
        elif SUBG_AUCV_RC == 1:
            if cf > 0 and rc_t > rcrautc_tmp * cf and ldmicro:
                hlc_hcf = cf
                hlc_lcf = 0
                hlc_hrc = rc_t
                hlc_lrc = 0

            elif cf > 0 and rc_t > C_RTMIN and ldmicro:
                hlc_hcf = 0
                hlc_lcf = cf
                hlc_hrc = 0
                hlc_lrc = rc_t

            else:
                hlc_hcf = 0
                hlc_lcf = 0
                hlc_hrc = 0
                hlc_lrc = 0

        # HSUBG_AUCV_RC = ADJU (2)
        elif SUBG_AUCV_RC == 2:
            sumrc_tmp = hlc_lrc + hlc_hrc if ldmicro else 0

            if sumrc_tmp > 0 and ldmicro:
                hlc_lrc *= rc_t / sumrc_tmp
                hlc_hrc *= rc_t / sumrc_tmp

            else:
                hlc_lrc = 0
                hlc_hrc = 0

        # HSUBG_AUCV_RC = PDF (3)
        elif SUBG_AUCV_RC == 3:

            # HSUBG_PR_PDF = SIGM (0)
            if SUBG_PR_PDF == 0:
                if rc_t > rcrautc_tmp + sigma_rc and ldmicro:
                    hlc_hcf = 1
                    hlc_lcf = 0
                    hlc_hrc = rc_t
                    hlc_lrc = 0

                elif (
                    rc_t > (rcrautc_tmp - sigma_rc)
                    and rc_t >= (rcrautc_tmp + sigma_rc)
                    and ldmicro
                ):
                    hlc_hcf = (rc_t + sigma_rc - rcrautc_tmp) / (2.0 * sigma_rc)
                    hlc_lcf = max(0.0, cf - hlc_hcf)
                    hlc_hrc = (
                        (rc_t + sigma_rc - rcrautc_tmp)
                        * (rc_t + sigma_rc + rcrautc_tmp)
                        / (4.0 * sigma_rc)
                    )
                    hlc_lrc = max(0.0, rc_t - hlc_hrc)

                elif rc_t > C_RTMIN and cf > 0 and ldmicro:
                    hlc_hcf = 0
                    hlc_lcf = cf
                    hlc_hrc = 0
                    hlc_lrc = rc_t

                else:
                    hlc_hcf = 0.0
                    hlc_lcf = 0.0
                    hlc_hrc = 0.0
                    hlc_lrc = 0.0

            # Translation note : l187 to l296 omitted since options are not used in AROME

    with computation(PARALLEL), interval(...):
        criauti_tmp = (
            min(CRIAUTI, 10 ** (ACRIAUTI * (t - TT) + BCRIAUTI)) if ldmicro else 0
        )

        # TODO: inline this code
        # HSUBG_AUCV_RI = NONE (0)
        if SUBG_AUCV_RI == 0:
            if ri_t > criauti_tmp and ldmicro:
                hli_hcf = 1
                hli_lcf = 0
                hli_hri = ri_t
                hli_lri = 0

            elif ri_t > I_RTMIN and ldmicro:
                hli_hcf = 0
                hli_lcf = 1
                hli_hri = 0
                hli_lri = ri_t

            else:
                hli_hcf = 0
                hli_lcf = 0
                hli_hri = 0
                hli_lri = 0

        # HSUBG_AUCV_RI = CLFR (1)
        elif SUBG_AUCV_RI == 1:
            if cf > 0 and ri_t > criauti_tmp * cf and ldmicro:
                hli_hcf = cf
                hli_hri = 0
                hli_hri = ri_t
                hli_lri = 0

            elif cf > 0 and ri_t > I_RTMIN and ldmicro:
                hli_hcf = 0
                hli_lcf = cf
                hli_hri = 0
                hli_lri = ri_t

            else:
                hli_hcf = 0
                hli_lcf = 0
                hli_hri = 0
                hli_lri = 0

        # HSUBG_AUCV_RI == 2
        elif SUBG_AUCV_RI == 2:
            sumri_tmp = hli_lri + hli_hri if ldmicro else 0

            if sumri_tmp > 0 and ldmicro:
                hli_lri *= ri_t / sumri_tmp
                hli_hri *= ri_t / sumri_tmp
            else:
                hli_lri = 0
                hli_hri = 0

    with computation(PARALLEL), interval(...):
        rf = max(hlc_hcf, hli_hcf) if ldmicro else 0


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=792, to_line=801
)
@stencil_collection("rain_fraction_sedimentation")
def rain_fraction_sedimentation(
    wr_r: Field["float"],
    wr_s: Field["float"],
    wr_g: Field["float"],
    rrs: Field["float"],
    rss: Field["float"],
    rgs: Field["float"],
):
    """Computes vertical rain fraction

    Args:
        wr_r (Field[float]): initial value for rain m.r.
        wr_s (Field[float]): initial value for snow m.r.
        wr_g (Field[float]): initial value for graupel m.r.
        rrs (Field[float]): tendency (source) for rain
        rss (Field[float]): tendency (source) for snow
        rgs (Field[float]): tendency (source) for graupel
    """

    from __externals__ import TSTEP

    with computation(PARALLEL), interval(0, 1):
        wr_r = rrs * TSTEP
        wr_s = rss * TSTEP
        wr_g = rgs * TSTEP


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90", from_line=792, to_line=801
)
@stencil_collection("ice4_rainfr_vert")
def ice4_rainfr_vert(
    prfr: Field["float"], rr: Field["float"], rs: Field["float"], rg: Field["float"]
):
    from __externals__ import S_RTMIN, R_RTMIN, G_RTMIN

    with computation(BACKWARD), interval(0, -1):
        if rr > R_RTMIN or rs > S_RTMIN or rg > G_RTMIN:

            prfr[0, 0, 0] = max(prfr[0, 0, 0], prfr[0, 0, 1])
            if prfr == 0:
                prfr = 1
        else:
            prfr = 0


@ported_method(
    from_file="PHYEX/src/common/micro/rain_ice.F90.func.h", from_line=816, to_line=830
)
@stencil_collection("fog_deposition")
def fog_deposition(
    rcs: Field["float"],
    rc_t: Field["float"],
    rhodref: Field["float"],
    dzz: Field["float"],
    inprc: Field[IJ, "float"],
):
    """Compute fog deposition on vegetation.
    Not activated in AROME.

    Args:
        rcs (Field[float]): source of cloud droplets
        rc_t (Field[float]): cloud droplets m.r.
        rhodref (Field[float]): dry density of air
        dzz (Field[float]): vertical spacing of cells
        inprc (Field[IJ, float]): deposition on vegetation
    """

    from __externals__ import VDEPOSC, RHOLW

    # Note : activated if LDEPOSC is True in rain_ice.F90
    with computation(FORWARD), interval(0, 1):
        rcs -= VDEPOSC * rc_t / dzz
        inprc[0, 0] += VDEPOSC * rc_t[0, 0, 0] * rhodref[0, 0, 0] / RHOLW


