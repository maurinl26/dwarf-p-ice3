## Documentation


dwarf-ice3-gt4py is a translation of Arome's microphysical parametrizations to GT4PY framework.

### Dependencies

The dwarf-ice3-gt4py project is built on GT4Py and ifs-physics-common:

 - [gt4py](https://github.com/GridTools/gt4py): implementation of HPC stencils using python DSL for GridTools.
 - [ifs-physics-common](https://github.com/stubbiali/ifs-physics-common): implementation of OOP overhead using ifs-physics-common which overload [sympl](https://sympl.readthedocs.io/en/latest/) for proper description of model components and fields.

### PHYEX

The project translates Arome's microphysics from [PHYEX](https://github.com/UMR-CNRM/PHYEX) tag [IAL_CY50T1](). Microphysical parametrization are splitted in two parts :

1. microphysical adjustments on vapour supersaturation,
2. computation of microphysical sources and sedimentation process.

Physical parametizations of Arome are described in _apl_arome.F90_. Entries for microphysical adjustments and sources computation are respectively _aro_adjust.F90_ and _aro_rain_ice.F90_.

### dwarf-p-cloudsc

This project follows the example of [cloudsc](https://github.com/ecmwf-ifs/dwarf-p-cloudsc) with its translation in GT4Py. 

### Translation notes

Since PHYEX parametrization are shared between Méso-NH, Arome and LMDZ, some parts of the source code have been omitted. The list of changes in translation between PHYEX microphysics and dwarf-ice3-gt4py can be found here : [translation notes](translation_options.md).


### Reference options

[Namelist](namelists.md) is the reference namelist of options translated and default parameters.