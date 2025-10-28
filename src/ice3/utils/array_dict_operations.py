# -*- coding: utf-8 -*-
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from numpy.typing import NDArray


############# Utils #############
def absolute_differences(
    fortran_fields: Dict[str, NDArray],
    gt4py_fields: Dict[str, NDArray],
    fields_to_compare: List[str]
) -> Dict[str, NDArray]:
    """Compute absolute difference on a list of fields

    Args:
        fortran_fields (NDArrayLikeDict): _description_
        gt4py_fields (NDArrayLikeDict): _description_
        fields_to_compare (_type_): _description_

    Returns:
        _type_: _description_
    """

    return {
        field_name : abs(gt4py_fields[field_name] - fortran_fields[field_name]).values.mean()
        if (gt4py_fields[field_name].shape == fortran_fields[field_name].shape)
        else None
        for field_name in fields_to_compare.keys()
    }


def remove_y_axis(fields: Dict[str, NDArray]):
    return {key: np.squeeze(array, axis=1) for key, array in fields.items()}


def unpack(fields: Dict[str, NDArray], domain: Tuple[int, int, int]) -> Dict[str, NDArray]:
    """Unpack as a 2d field

    Args:
        fortran_fields (_type_): _description_
        component_grid (_type_): _description_

    Returns:
        _type_: _description_
    """
    return {
        key: array.reshape(domain[0] * domain[1], domain[2])
        for key, array in fields.items()
    }
