import numpy as np
from numpy cimport ndarray
import cython

cimport ice_adjust_wrapper

@cython.boundscheck(False)
@cython.wraparound(False)
def ice_adjust_wrapper(
    int d_nijt,
    np.ndarray[double, ndim=2, mode='fortran'] val,

):

    # cdef output fields
    cdef np.ndarray[double, ndim=2, mode='fortran'] two_val
    
    two_val = np.empty_like(val)
    
    ice_adjust_wrapper.ice_adjust_wrapper(

    )

    

