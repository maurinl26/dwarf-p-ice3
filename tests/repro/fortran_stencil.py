from pathlib import Path
import logging
import fmodpy

fortran_script = "mode_ice4_warm.F90"
CURRENT_DIRECTORY = Path.cwd()
ROOT_DIRECTORY = CURRENT_DIRECTORY
STENCILS_DIRECTORY = Path(
            ROOT_DIRECTORY, "src", "ice3_gt4py", "stencils_fortran"
        )


class FortranStencil:

    def __init__(self,
                 script: str,
                 module: str,
                 subourtine: str):
        
        script_path = Path(STENCILS_DIRECTORY, fortran_script)
        logging.info(f"Fortran script path {script_path}")
        fortran_script = fmodpy.fimport(script_path)
        
        self.fortran_script = fmodpy.fimport(script_path)
        self.module = setattr(self.fortran_script, "")
        
        