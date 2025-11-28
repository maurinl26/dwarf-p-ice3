import fmodpy
from pathlib import Path
import logging

def compile_fortran_stencil(
    fortran_script: str, fortran_module: str, fortran_stencil: str
):
    """Compile and import a Fortran stencil using fmodpy.
    
    This function dynamically compiles a Fortran subroutine from the stencils_fortran
    directory and returns a callable Python wrapper. The Fortran code is located,
    imported via fmodpy, and the specified module and stencil function are extracted
    for use in Python.

    Args:
        fortran_script (str): Name of the Fortran source file (e.g., "ice4_nucleation.F90")
            located in the stencils_fortran directory
        fortran_module (str): Name of the Fortran module within the script that contains
            the stencil subroutine
        fortran_stencil (str): Name of the specific stencil subroutine to extract from
            the module

    Returns:
        callable: The compiled Fortran subroutine as a callable Python function that can
            be invoked with appropriate arguments
    """
    #### Fortran subroutine
    root_directory = Path(__file__).parent.parent
    stencils_directory = Path(root_directory, "stencils_fortran")
    script_path = Path(stencils_directory, fortran_script)

    logging.info(f"Fortran script path {script_path}")
    fortran_script = fmodpy.fimport(script_path)
    mode = getattr(fortran_script, fortran_module)
    return getattr(mode, fortran_stencil)
