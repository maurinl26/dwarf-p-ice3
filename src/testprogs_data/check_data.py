# -*- coding: utf-8 -*-
import netCDF4 as nc


if __name__ == "__main__":

    ds = nc.Dataset("./testprogs_data/data/ice_adjust/reference.nc")

    print(ds["PRHODREF"][...])
    print(ds["PRHODREF"])
    print(ds["PRHODREF"][...].shape)

    print(ds["IJ"][...])
    print(ds["K"][...])
    print(ds.variables)
    print(ds.dimensions)
