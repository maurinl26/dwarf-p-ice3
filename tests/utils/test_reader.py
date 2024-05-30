# -*- coding: utf-8 -*-
from ice3_gt4py.utils.reader import NetCDFReader


if __name__ == "__main__":

    reader = NetCDFReader("./data/rain_ice/reference.nc")

    ldmicro = reader.get_field("LLMICRO").astype(bool)

    print(ldmicro.shape)
    print(ldmicro.dtype)

    ldmicro_bool = ldmicro.astype(bool)
    print(ldmicro_bool.dtype)

    sea = reader.get_field("PSEA")
    print(sea.shape)
