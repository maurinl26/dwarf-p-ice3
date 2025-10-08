cmake_minimum_required( VERSION 3.15.0 FATAL_ERROR )

find_package( ecbuild 3.1 REQUIRED )

set(BUILD_SHARED_LIBS ${IFS_SHARED_LIBS})

set( PROJECT_NAME phyex )

project( ${PROJECT_NAME} VERSION 1.0 LANGUAGES Fortran C CXX )

include(phyex_stencils)

### finalize project
ecbuild_install_project( NAME ${PROJECT_NAME} )

ecbuild_print_summary()


