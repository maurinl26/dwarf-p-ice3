# -*- coding: utf-8 -*-
from enum import Enum


class Processes(Enum):
    RCHONI = 1
    RVDEPS = 2
    RIAGGS = 3  # Aggregation on r_s
    RIAUTS = 4  # Autoconversion of r_i for r_s production
    RVDEPG = 5  # Deposition on r_g
    RCAUTR = 6  # Autoconversion of r_c for r_r production
    RCACCR = 7  # Accretion of r_c for r_r production
    RREVAV = 8  # Evaporation of r_r
    RCBERI = 9  # Bergeron-Findeisen effect
    RHMLTR = 10  # Melting of the hailstones
    RSMLTG = 11  # Conversion-Melting of the aggregates
    RCMLTSR = 12  # Cloud droplet collection onto aggregates by positive temperature
    RRACCSS = 13
    RRACCSG = 14
    RSACCRG = 15  # Rain accretion onto the aggregates
    RCRIMSS = 16
    RCRIMSG = 17
    RSRIMCG = 18  # Cloud droplet riming of the aggregates
    RICFRRG = 19
    RRCFRIG = 20
    RICFRR = 21  # Rain contact freezing
    RCWETG = 22
    RIWETG = 23
    RRWETG = 24
    RSWETG = 25  # Graupel wet growth
    RCDRYG = 26
    RIDRYG = 27
    RRDRYG = 28
    RSDRYG = 29  # Graupel dry growth
    RWETGH = 30  # Conversion of graupel nto hail
    RGMLTR = 31  # Melting of the graupel
    RCWETH = 32
    RIWETH = 33
    RSWETH = 34
    RGWETH = 35
    RRWETH = 36  # Dry growth of hailstone
    RCDRYH = 37
    RIDRYH = 38
    RSDRYH = 39
    RRDRYH = 40
    RGDRYH = 41  # Wet growth of hailstone
    RDRYHG = 42

    # tendencies computed only with a mixing ratio change
    RVHENI_MR = 43  # heterogeneous nucleation mixing ratio change
    RRHONG_MR = 44  # Spontaneous freezing mixing ratio change
    RIMLTC_MR = 45  # Cloud ce melting mixing ratio change

    # Extra term computed as a mixing ratio change to be added to other term
    RSRIMCG_MR = 46  # Cloud droplet riming of the aggregates
    RWETGH_MR = 47
