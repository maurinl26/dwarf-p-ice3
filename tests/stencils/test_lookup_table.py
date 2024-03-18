# -*- coding: utf-8 -*-
from ifs_physics_common.framework.grid import Grid

if __name__ == "__main__":

    ncolx, ncoly = 40, 40

    # grille 2D
    lookup_grid = Grid((ncolx, ncoly), ("xcol", "ycol"))

    # instanciate field for actual lookup tables
    # RACCSS
    # RACCS
    # RACCRG
