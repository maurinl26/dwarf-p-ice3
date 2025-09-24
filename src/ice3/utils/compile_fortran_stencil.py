from pathlib import Path
import logging
import fmodpy


def compile_fortran_stencil(
    fortran_script: str, fortran_module: str, fortran_stencil: str
):
    """Compile fortran stencil wrapped in a fortran file + module with fmodpy

    Args:
        fortran_script (str): _description_
        fortran_module (str): _description_
        fortran_stencil (str): _description_

    Returns:
        _type_: _description_
    """
    #### Fortran subroutine
    root_directory = Path(__file__).parent.parent
    stencils_directory = Path(root_directory, "src", "ice3_gt4py", "stencils_fortran")
    script_path = Path(stencils_directory, fortran_script)

    logging.info(f"Fortran script path {script_path}")
    fortran_script = fmodpy.fimport(script_path)
    mode = getattr(fortran_script, fortran_module)
    return getattr(mode, fortran_stencil)