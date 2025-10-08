ecbuild_info("[phyex_stencils]")

ecbuild_add_library(TARGET phyex_stencils
    LINKER_LANGUAGE Fortran
    SOURCES_GLOB
      src/phyex_stencils/ice_adjust/*.F90
    PUBLIC_LIBS
    PUBLIC_INCLUDES
    )

