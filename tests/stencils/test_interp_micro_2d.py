import cupy as cp
import numpy as np
import numpy.typing as npt

def compute_accretion_interp(
    lambda_s: npt.ArrayLike,
    lambda_r: npt.ArrayLike,
    plbdas: npt.ArrayLike,
    gacc: npt.ArrayLike,
    ldsoft: bool,
    ACCINTP1S: np.float32,
    ACCINTP2S: np.float32,
    NACCLBDAS: np.int32,
    ACCINTP1R: np.float32,
    ACCINTP2R: np.float32,
    NACCLBDAR: np.int32,
    ker_raccss: npt.ArrayLike,
    ker_raccs: npt.ArrayLike,
    ker_saccrg: npt.ArrayLike
):
    """ Bilinear interpolation from ice4_fast_rs """

    # index micro2d acc s
    index_s = np.clip(
                ACCINTP1S * np.log(lambda_s) + ACCINTP2S,
                NACCLBDAS - 0.00001,
                1.00001
                )
        
    idx_s = np.floor(index_s)
    idx_s2 = idx_s + 1
    weight_s = index_s - idx_s

                
    # index micro2d acc r
    index_r = np.clip(
            ACCINTP1R * np.log(lambda_r) + ACCINTP2R,
            NACCLBDAR - 0.00001,
            1.00001
            )
    idx_r = np.floor(index_r)
    idx_r2 = idx_r + 1
    weight_r = index_r - idx_r
 
    # Bilinear interpolation for RACCSS kernel
    zzw1 = (
                (
                   weight_r * ker_raccss.take(idx_s2, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_raccss.take(idx_s2, axis=0).take(idx_r)      
                ) * weight_s
                + (
                  weight_r * ker_raccss.take(idx_s, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_raccss.take(idx_s, axis=0).take(idx_r)             
                ) * (1 - weight_s)
        )
           
    # Bilinear interpolation for RACCS kernel
    zzw2 = (
                (
                   weight_r * ker_raccs.take(idx_s2, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_raccs.take(idx_s2, axis=0).take(idx_r)      
                ) * weight_s
                + (
                  weight_r * ker_raccs.take(idx_s, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_raccs.take(idx_s, axis=0).take(idx_r)             
                ) * (1 - weight_s)
        )
        
     
    # Bilinear interpolation for SACCRG kernel
    zzw3 = (
                (
                   weight_r * ker_saccrg.take(idx_s2, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_saccrg.take(idx_s2, axis=0).take(idx_r)      
                ) * weight_s
                + (
                  weight_r * ker_saccrg.take(idx_s, axis=0).take(idx_r2) 
                + (1 - weight_r) * ker_saccrg.take(idx_s, axis=0).take(idx_r)             
                ) * (1 - weight_s)
    )
    
       