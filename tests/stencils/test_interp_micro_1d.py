
import cupy as cp
import numpy as np
import numpy.typing as npt

def interp_micro_1d(
        lambda_s: npt.ArrayLike,
        ker_gaminc_rim1: npt.ArrayLike,
        ker_gaminc_rim2: npt.ArrayLike,
        ker_gaminc_rim4: npt.ArrayLike,
        RIMINTP1: np.float32,
        RIMINTP2: np.float32,
        NGAMINC: np.int32,
):
    """ Linear interpolation """

    index = np.clip(
        RIMINTP1 * np.log(lambda_s) + RIMINTP2, 
        NGAMINC - 0.00001,
        1.00001) 

    idx_interp = np.floor(index)
    idx_interp_2 = idx_interp + 1
    weight = index - idx_interp

    zzw1 = weight * ker_gaminc_rim1.take(idx_interp) \
        (1 - weight) * ker_gaminc_rim1.take(idx_interp_2) 
    

    zzw2 = weight * ker_gaminc_rim2.take(idx_interp) \
        (1 - weight) * ker_gaminc_rim2.take(idx_interp_2) 
    
    zzw4 = weight * ker_gaminc_rim4.take(idx_interp) \
        (1 - weight) * ker_gaminc_rim4.take(idx_interp_2) 
    
    return zzw1, zzw2, zzw4

   

