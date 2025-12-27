cmake_minimum_required(VERSION 3.12)

# Set gfortran as the default Fortran compiler
if(NOT DEFINED CMAKE_Fortran_COMPILER)
    set(CMAKE_Fortran_COMPILER "gfortran")
endif()

project(PHYEX_ICE_ADJUST C Fortran)

# ============================================================================
# OPENACC GPU ACCELERATION OPTION
# ============================================================================
option(ENABLE_OPENACC "Enable OpenACC GPU acceleration (requires NVIDIA HPC SDK)" OFF)

# Set Fortran compiler flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O2 -fPIC -cpp")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fbacktrace -fcheck=all")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")

# Add OpenACC flags if enabled
if(ENABLE_OPENACC)
    message(STATUS "OpenACC GPU acceleration ENABLED")

    # Check if using NVIDIA HPC SDK compiler (nvfortran)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "^(NVHPC|PGI)$")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -acc -Minfo=accel -gpu=managed -Minline")
        set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -fast -Mcuda")
        message(STATUS "Using NVIDIA HPC SDK compiler with OpenACC")
    else()
        message(WARNING "OpenACC enabled but not using NVIDIA HPC SDK compiler. GPU acceleration may not work.")
        message(WARNING "Current compiler: ${CMAKE_Fortran_COMPILER_ID}")
        message(WARNING "Set FC=nvfortran to use NVIDIA HPC SDK")
    endif()
else()
    message(STATUS "OpenACC GPU acceleration DISABLED (use -DENABLE_OPENACC=ON to enable)")
endif()

# Set module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

# macOS specific linker flags for Python extensions
if(APPLE)
    # Flang driver doesn't recognize macOS linker flags directly, pass them via -Wl,
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")
    
    # Override CMake defaults for shared library creation to use -shared (standard LLVM/Clang flag)
    set(CMAKE_SHARED_LIBRARY_CREATE_Fortran_FLAGS "-shared -Wl,-headerpad_max_install_names")
    # Use -Xlinker to ensure the path following -install_name is treated as a linker argument, not an input file
    set(CMAKE_SHARED_LIBRARY_SONAME_Fortran_FLAG "-Xlinker -install_name -Xlinker")
endif()

# Define PHYEX source directory
set(PHYEX_DIR ${CMAKE_SOURCE_DIR}/PHYEX-IAL_CY50T1)

# Add stub include directory
include_directories(${PHYEX_DIR}/stubs/include)

# ============================================================================
# STUB MODULES (for external IFS dependencies)
# ============================================================================
set(STUB_SOURCES
    ${PHYEX_DIR}/stubs/parkind1.F90
    ${PHYEX_DIR}/stubs/ec_parkind.F90
    ${PHYEX_DIR}/stubs/yomhook.F90
    ${PHYEX_DIR}/stubs/yomlun.F90
    ${PHYEX_DIR}/stubs/abor1.F90
    ${PHYEX_DIR}/stubs/fmlook_ll.F90
    ${PHYEX_DIR}/stubs/posnam.F90
)

# ============================================================================
# AUXILIARY MODULES (modd_* - data structures, no dependencies)
# ============================================================================
set(AUX_MODD_SOURCES
    ${PHYEX_DIR}/aux/modd_precision.F90
    ${PHYEX_DIR}/aux/modd_parameters.F90
    ${PHYEX_DIR}/aux/modd_cst.F90
    ${PHYEX_DIR}/aux/modd_dimphyexn.F90
    ${PHYEX_DIR}/aux/modd_field.F90
    ${PHYEX_DIR}/aux/modd_io.F90
    ${PHYEX_DIR}/aux/modd_argslist_ll.F90
    ${PHYEX_DIR}/aux/modd_budget.F90
    ${PHYEX_DIR}/aux/modd_conf.F90
    ${PHYEX_DIR}/aux/modd_les.F90
    ${PHYEX_DIR}/aux/modd_misc.F90
    ${PHYEX_DIR}/aux/modd_nsv.F90
)

# ============================================================================
# MICROPHYSICS MODULES (modd_* from micro/)
# ============================================================================
set(MICRO_MODD_SOURCES
    ${PHYEX_DIR}/micro/modd_tiwmx.F90
    ${PHYEX_DIR}/micro/modd_rain_ice_descrn.F90
    ${PHYEX_DIR}/micro/modd_rain_ice_paramn.F90
    ${PHYEX_DIR}/micro/modd_param_icen.F90
    ${PHYEX_DIR}/micro/modd_nebn.F90
    ${PHYEX_DIR}/micro/modd_cloudparn.F90
    ${PHYEX_DIR}/micro/modd_fields_address.F90
)

# ============================================================================
# OPENACC GPU-ACCELERATED MODULES (conditional)
# ============================================================================
if(ENABLE_OPENACC)
    set(MICRO_MODD_SOURCES_ACC
        ${PHYEX_DIR}/micro/modd_tiwmx_acc.F90
        ${PHYEX_DIR}/micro/modd_nebn_acc.F90
    )

    set(MICRO_MODE_SOURCES_ACC
        ${PHYEX_DIR}/micro/mode_tiwmx_acc.F90
    )

    set(MAIN_SOURCES_ACC
        ${PHYEX_DIR}/micro/condensation_acc.F90
        ${PHYEX_DIR}/micro/ice_adjust_acc.F90
    )

    set(BRIDGE_SOURCES_ACC
        ${PHYEX_DIR}/bridge/phyex_bridge_acc.F90
    )
else()
    set(MICRO_MODD_SOURCES_ACC "")
    set(MICRO_MODE_SOURCES_ACC "")
    set(MAIN_SOURCES_ACC "")
    set(BRIDGE_SOURCES_ACC "")
endif()

# ============================================================================
# TURBULENCE MODULES (modd_* from turb/)
# ============================================================================
set(TURB_MODD_SOURCES
    ${PHYEX_DIR}/turb/modd_cturb.F90
    ${PHYEX_DIR}/turb/modd_turbn.F90
    ${PHYEX_DIR}/turb/modd_param_mfshalln.F90
)

# ============================================================================
# CONVECTION MODULES (modd_* from conv/)
# ============================================================================
set(CONV_MODD_SOURCES
    ${PHYEX_DIR}/conv/modd_convpar.F90
    ${PHYEX_DIR}/conv/modd_convpar_shal.F90
    ${PHYEX_DIR}/conv/modd_convparext.F90
)

# ============================================================================
# PHYEX AGGREGATION MODULE (depends on all other modules)
# ============================================================================
set(PHYEX_SOURCES
    ${PHYEX_DIR}/aux/modd_phyex.F90
)

# ============================================================================
# AUXILIARY MODE MODULES (mode_* - functions, depend on modd_*)
# ============================================================================
set(AUX_MODE_SOURCES
    ${PHYEX_DIR}/aux/mode_msg.F90
    ${PHYEX_DIR}/aux/mode_mppdb.F90
    ${PHYEX_DIR}/aux/mode_ll.F90
    ${PHYEX_DIR}/aux/mode_thermo.F90
    ${PHYEX_DIR}/aux/mode_check_nam_val.F90
    ${PHYEX_DIR}/aux/mode_fill_dimphyexn.F90
    ${PHYEX_DIR}/aux/mode_posnam_phy.F90
    ${PHYEX_DIR}/aux/mode_shuman_phy.F90
    ${PHYEX_DIR}/aux/mode_sources_neg_correct.F90
    ${PHYEX_DIR}/aux/mode_ini_cst.F90
    ${PHYEX_DIR}/aux/mode_io_field_write.F90
    ${PHYEX_DIR}/aux/mode_io_field_write_phy.F90
    ${PHYEX_DIR}/aux/mode_argslist_ll_phy.F90
    ${PHYEX_DIR}/aux/mode_gradient_m_phy.F90
    ${PHYEX_DIR}/aux/mode_gradient_u_phy.F90
    ${PHYEX_DIR}/aux/mode_gradient_v_phy.F90
    ${PHYEX_DIR}/aux/mode_gradient_w_phy.F90
    ${PHYEX_DIR}/aux/mode_mnh_zwork.F90
)

# ============================================================================
# AUXILIARY FUNCTIONS AND SUBROUTINES
# ============================================================================
set(AUX_FUNC_SOURCES
    ${PHYEX_DIR}/aux/shuman.F90
    ${PHYEX_DIR}/aux/gamma.F90
    ${PHYEX_DIR}/aux/gamma_inc.F90
    ${PHYEX_DIR}/aux/general_gamma.F90
    ${PHYEX_DIR}/aux/second_mnh.F90
    ${PHYEX_DIR}/aux/gradient_m.F90
    ${PHYEX_DIR}/aux/gradient_u.F90
    ${PHYEX_DIR}/aux/gradient_v.F90
    ${PHYEX_DIR}/aux/gradient_w.F90
    ${PHYEX_DIR}/aux/ini_phyex.F90
    ${PHYEX_DIR}/aux/tools.F90
    ${PHYEX_DIR}/aux/bitrep.F90
)

# ============================================================================
# MICROPHYSICS MODE MODULES
# ============================================================================
set(MICRO_MODE_SOURCES
    ${PHYEX_DIR}/micro/mode_tiwmx.F90
    ${PHYEX_DIR}/micro/mode_tiwmx_fun.F90
    ${PHYEX_DIR}/micro/mode_tiwmx_tab.F90
    ${PHYEX_DIR}/micro/mode_qsatmx_tab.F90
    ${PHYEX_DIR}/micro/mode_icecloud.F90
    ${PHYEX_DIR}/micro/mode_ini_tiwmx.F90
    ${PHYEX_DIR}/micro/mode_rrcolss.F90
    ${PHYEX_DIR}/micro/mode_rzcolx.F90
    ${PHYEX_DIR}/micro/mode_rscolrg.F90
    ${PHYEX_DIR}/micro/mode_read_xker_raccs.F90
    ${PHYEX_DIR}/micro/mode_read_xker_sdryg.F90
    ${PHYEX_DIR}/micro/mode_read_xker_rdryg.F90
    ${PHYEX_DIR}/micro/mode_read_xker_sweth.F90
    ${PHYEX_DIR}/micro/mode_read_xker_gweth.F90
    ${PHYEX_DIR}/micro/mode_read_xker_rweth.F90
    ${PHYEX_DIR}/micro/mode_ini_rain_ice.F90
    ${PHYEX_DIR}/micro/mode_ice4_rainfr_vert.F90
    ${PHYEX_DIR}/micro/mode_ice4_compute_pdf.F90
    ${PHYEX_DIR}/micro/mode_ice4_sedimentation_stat.F90
    ${PHYEX_DIR}/micro/mode_ice4_pack.F90
    ${PHYEX_DIR}/micro/mode_ice4_correct_negativities.F90
    ${PHYEX_DIR}/micro/mode_ice4_budgets.F90
    ${PHYEX_DIR}/micro/mode_ice4_fast_rg.F90
    ${PHYEX_DIR}/micro/mode_ice4_fast_rh.F90
    ${PHYEX_DIR}/micro/mode_ice4_fast_ri.F90
    ${PHYEX_DIR}/micro/mode_ice4_fast_ri_rs.F90
    ${PHYEX_DIR}/micro/mode_ice4_fast_rs.F90
    ${PHYEX_DIR}/micro/mode_ice4_rimltc.F90
    ${PHYEX_DIR}/micro/mode_ice4_rrhong.F90
    ${PHYEX_DIR}/micro/mode_ice4_slow.F90
    ${PHYEX_DIR}/micro/mode_ice4_stepping.F90
    ${PHYEX_DIR}/micro/mode_ice4_tendencies.F90
    ${PHYEX_DIR}/micro/mode_ice4_warm.F90
    ${PHYEX_DIR}/turb/mode_ini_mfshall.F90
    ${PHYEX_DIR}/turb/mode_ini_turb.F90
)

# ============================================================================
# MICROPHYSICS FUNCTIONS
# ============================================================================
set(MICRO_FUNC_SOURCES
    ${PHYEX_DIR}/micro/hypser.f90
    ${PHYEX_DIR}/micro/hypgeo.F90
    ${PHYEX_DIR}/micro/momg.F90
)

# ============================================================================
# CONVECTION SUBROUTINES
# ============================================================================
set(CONV_SOURCES
    ${PHYEX_DIR}/conv/convect_satmixratio.F90
    ${PHYEX_DIR}/conv/convect_condens.F90
    ${PHYEX_DIR}/conv/convect_mixing_funct.F90
    ${PHYEX_DIR}/conv/convect_trigger_shal.F90
    ${PHYEX_DIR}/conv/convect_updraft_shal.F90
    ${PHYEX_DIR}/conv/convect_closure_thrvlcl.F90
    ${PHYEX_DIR}/conv/convect_closure_adjust_shal.F90
    ${PHYEX_DIR}/conv/convect_closure_shal.F90
    ${PHYEX_DIR}/conv/convect_chem_transport.F90
    ${PHYEX_DIR}/conv/shallow_convection_part1.F90
    ${PHYEX_DIR}/conv/shallow_convection_part2.F90
    ${PHYEX_DIR}/conv/shallow_convection_part2_select.F90
    ${PHYEX_DIR}/conv/shallow_convection.F90
)

# ============================================================================
# MODI INTERFACES
# ============================================================================
set(MODI_SOURCES
    ${PHYEX_DIR}/aux/modi_gamma.F90
    ${PHYEX_DIR}/aux/modi_gamma_inc.F90
    ${PHYEX_DIR}/aux/modi_general_gamma.F90
    ${PHYEX_DIR}/aux/modi_second_mnh.F90
    ${PHYEX_DIR}/aux/modi_shuman.F90
    ${PHYEX_DIR}/aux/modi_gradient_m.F90
    ${PHYEX_DIR}/aux/modi_gradient_u.F90
    ${PHYEX_DIR}/aux/modi_gradient_v.F90
    ${PHYEX_DIR}/aux/modi_gradient_w.F90
    ${PHYEX_DIR}/aux/modi_ini_phyex.F90
    ${PHYEX_DIR}/micro/modi_ice_adjust.F90
    ${PHYEX_DIR}/micro/modi_condensation.F90
    ${PHYEX_DIR}/micro/modi_rain_ice.F90
    ${PHYEX_DIR}/conv/modi_shallow_convection_part1.F90
    ${PHYEX_DIR}/conv/modi_shallow_convection_part2.F90
    ${PHYEX_DIR}/conv/modi_shallow_convection_part2_select.F90
    ${PHYEX_DIR}/conv/modi_shallow_convection.F90
)

# ============================================================================
# MAIN MICROPHYSICS AND CONVECTION ROUTINES
# ============================================================================
set(MAIN_SOURCES
    ${PHYEX_DIR}/micro/condensation.F90
    ${PHYEX_DIR}/micro/ice_adjust.F90
    ${PHYEX_DIR}/micro/rain_ice.F90
)

# ============================================================================
# BRIDGE MODULE (C-callable wrapper for Python/Cython)
# ============================================================================
set(BRIDGE_SOURCES
    ${PHYEX_DIR}/bridge/phyex_bridge.F90
)

# ============================================================================
# COMBINE ALL SOURCES IN PROPER ORDER
# ============================================================================
set(ALL_SOURCES
    # Level 0: External dependency stubs (must be first!)
    ${STUB_SOURCES}

    # Level 1: Basic modules without dependencies
    ${AUX_MODD_SOURCES}

    # Level 2: Microphysics, turbulence, and convection data modules
    ${MICRO_MODD_SOURCES}
    ${TURB_MODD_SOURCES}
    ${CONV_MODD_SOURCES}

    # Level 2b: OpenACC GPU-accelerated data modules (if enabled)
    ${MICRO_MODD_SOURCES_ACC}

    # Level 3: PHYEX aggregation module (depends on all modd_* modules)
    ${PHYEX_SOURCES}

    # Level 4: MODI interfaces
    ${MODI_SOURCES}

    # Level 5: Auxiliary mode modules
    ${AUX_MODE_SOURCES}

    # Level 6: Auxiliary functions
    ${AUX_FUNC_SOURCES}

    # Level 7: Microphysics mode modules
    ${MICRO_MODE_SOURCES}

    # Level 7b: OpenACC GPU-accelerated mode modules (if enabled)
    ${MICRO_MODE_SOURCES_ACC}

    # Level 8: Microphysics functions
    ${MICRO_FUNC_SOURCES}

    # Level 9: Main routines
    ${MAIN_SOURCES}

    # Level 9b: OpenACC GPU-accelerated main routines (if enabled)
    ${MAIN_SOURCES_ACC}

    # Level 9c: Convection subroutines (shallow_convection and dependencies)
    ${CONV_SOURCES}

    # Level 10: Bridge module (depends on everything)
    ${BRIDGE_SOURCES}

    # Level 10b: OpenACC GPU bridge (if enabled)
    ${BRIDGE_SOURCES_ACC}
)

# ============================================================================
# CREATE LIBRARY
# ============================================================================

# Create a shared library for Python integration
add_library(ice_adjust_phyex SHARED ${ALL_SOURCES})

# Set library properties
set_target_properties(ice_adjust_phyex PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PREFIX "lib"
)

# ============================================================================
# CYTHON EXTENSION MODULE
# ============================================================================

# Find Python and Cython
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# ============================================================================
# PHYEX STRUCTURES MODULE (Cython cclasses for parameter structures)
# ============================================================================
# NOTE: phyex_structures.pyx is not present in the project, commented out
# set(PHYEX_STRUCTURES_SOURCE ${CMAKE_SOURCE_DIR}/src/ice3/phyex_structures.pyx)
# set(PHYEX_STRUCTURES_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/phyex_structures.c)

# add_custom_command(
#     OUTPUT ${PHYEX_STRUCTURES_OUTPUT}
#     COMMAND ${Python_EXECUTABLE} -m cython
#         ${PHYEX_STRUCTURES_SOURCE}
#         -o ${PHYEX_STRUCTURES_OUTPUT}
#     DEPENDS ${PHYEX_STRUCTURES_SOURCE}
#     COMMENT "Cythonizing phyex_structures.pyx to C"
# )

# # Create Python extension module for phyex structures
# Python_add_library(phyex_structures MODULE ${PHYEX_STRUCTURES_OUTPUT})

# # Explicitly set the language to C
# set_source_files_properties(${PHYEX_STRUCTURES_OUTPUT} PROPERTIES LANGUAGE C)

# # Include NumPy headers
# target_include_directories(phyex_structures PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# # Set output name (remove lib prefix for Python modules)
# set_target_properties(phyex_structures PROPERTIES
#     PREFIX ""
#     OUTPUT_NAME "phyex_structures"
#     LINKER_LANGUAGE C
# )

# CPU Cython wrapper (always built)
set(CYTHON_SOURCE ${PHYEX_DIR}/bridge/_phyex_wrapper.pyx)
set(CYTHON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_phyex_wrapper.c)

add_custom_command(
    OUTPUT ${CYTHON_OUTPUT}
    COMMAND ${Python_EXECUTABLE} -m cython
        ${CYTHON_SOURCE}
        -o ${CYTHON_OUTPUT}
    DEPENDS ${CYTHON_SOURCE}
    COMMENT "Cythonizing _phyex_wrapper.pyx to C"
)

# Create Python extension module from generated C code
Python_add_library(_phyex_wrapper MODULE ${CYTHON_OUTPUT})

# Explicitly set the language to C
set_source_files_properties(${CYTHON_OUTPUT} PROPERTIES LANGUAGE C)

# Link with the Fortran library
target_link_libraries(_phyex_wrapper PRIVATE ice_adjust_phyex)

# Include NumPy headers
target_include_directories(_phyex_wrapper PRIVATE ${Python_NumPy_INCLUDE_DIRS})

# Set output name (remove lib prefix for Python modules)
set_target_properties(_phyex_wrapper PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_phyex_wrapper"
    LINKER_LANGUAGE C
)

# GPU Cython wrapper (conditional on OpenACC)
if(ENABLE_OPENACC)
    message(STATUS "Building GPU-accelerated Cython wrapper (_phyex_wrapper_acc)")

    set(CYTHON_SOURCE_ACC ${PHYEX_DIR}/bridge/_phyex_wrapper_acc.pyx)
    set(CYTHON_OUTPUT_ACC ${CMAKE_CURRENT_BINARY_DIR}/_phyex_wrapper_acc.c)

    add_custom_command(
        OUTPUT ${CYTHON_OUTPUT_ACC}
        COMMAND ${Python_EXECUTABLE} -m cython
            ${CYTHON_SOURCE_ACC}
            -o ${CYTHON_OUTPUT_ACC}
        DEPENDS ${CYTHON_SOURCE_ACC}
        COMMENT "Cythonizing _phyex_wrapper_acc.pyx to C (GPU version)"
    )

    # Create GPU Python extension module
    Python_add_library(_phyex_wrapper_acc MODULE ${CYTHON_OUTPUT_ACC})

    # Explicitly set the language to C
    set_source_files_properties(${CYTHON_OUTPUT_ACC} PROPERTIES LANGUAGE C)

    # Link with the Fortran library
    target_link_libraries(_phyex_wrapper_acc PRIVATE ice_adjust_phyex)

    # Include NumPy headers
    target_include_directories(_phyex_wrapper_acc PRIVATE ${Python_NumPy_INCLUDE_DIRS})

    # Try to find CuPy include directory
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import cupy; print(cupy.get_include())"
        OUTPUT_VARIABLE CUPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(CUPY_INCLUDE_DIR)
        message(STATUS "Found CuPy include directory: ${CUPY_INCLUDE_DIR}")
        target_include_directories(_phyex_wrapper_acc PRIVATE ${CUPY_INCLUDE_DIR})
    else()
        message(WARNING "CuPy not found - GPU wrapper may not work. Install with: pip install cupy-cuda12x")
    endif()

    # Set output name
    set_target_properties(_phyex_wrapper_acc PROPERTIES
        PREFIX ""
        OUTPUT_NAME "_phyex_wrapper_acc"
        LINKER_LANGUAGE C
    )
endif()

# ============================================================================
# INSTALLATION RULES
# ============================================================================

# Installation rules for scikit-build-core
install(TARGETS ice_adjust_phyex
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
)

# Install phyex_structures module (commented out - module not built)
# install(TARGETS phyex_structures
#     LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
# )

# Install CPU Cython extension module
install(TARGETS _phyex_wrapper
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
)

# Install GPU Cython extension module (if built)
if(ENABLE_OPENACC)
    install(TARGETS _phyex_wrapper_acc
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    )
endif()

install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
    DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3/include
    FILES_MATCHING PATTERN "*.mod"
)

# Also support traditional installation
if(NOT SKBUILD)
    install(TARGETS ice_adjust_phyex
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    # install(TARGETS phyex_structures
    #     LIBRARY DESTINATION lib/python/ice3
    # )

    install(TARGETS _phyex_wrapper
        LIBRARY DESTINATION lib/python/ice3
    )

    # Install GPU wrapper if built
    if(ENABLE_OPENACC)
        install(TARGETS _phyex_wrapper_acc
            LIBRARY DESTINATION lib/python/ice3
        )
    endif()

    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
        DESTINATION include/ice_adjust_phyex
        FILES_MATCHING PATTERN "*.mod"
    )
endif()

# ============================================================================
# OPTIONAL: Create a test executable
# ============================================================================
option(BUILD_TEST_EXECUTABLE "Build test executable" OFF)

if(BUILD_TEST_EXECUTABLE)
    add_executable(test_ice_adjust test_main.F90)
    target_link_libraries(test_ice_adjust ice_adjust_phyex)
endif()

# ============================================================================
# PRINT CONFIGURATION SUMMARY
# ============================================================================
message(STATUS "====================================")
message(STATUS "PHYEX ICE_ADJUST Configuration")
message(STATUS "====================================")
message(STATUS "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_Fortran_COMPILER_ID}")
message(STATUS "Compiler Flags: ${CMAKE_Fortran_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PHYEX Directory: ${PHYEX_DIR}")
message(STATUS "Module Directory: ${CMAKE_Fortran_MODULE_DIRECTORY}")
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "NumPy Include: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "Cython Modules: phyex_structures, _phyex_wrapper")

if(ENABLE_OPENACC)
    message(STATUS "")
    message(STATUS "GPU ACCELERATION: ENABLED")
    message(STATUS "  OpenACC: YES")
    message(STATUS "  GPU Fortran modules: ${MAIN_SOURCES_ACC}")
    message(STATUS "  GPU Cython wrapper: _phyex_wrapper_acc")
    if(CUPY_INCLUDE_DIR)
        message(STATUS "  CuPy: ${CUPY_INCLUDE_DIR}")
    else()
        message(STATUS "  CuPy: NOT FOUND (install: pip install cupy-cuda12x)")
    endif()
else()
    message(STATUS "")
    message(STATUS "GPU ACCELERATION: DISABLED")
    message(STATUS "  Use -DENABLE_OPENACC=ON to enable GPU support")
    message(STATUS "  Requires: NVIDIA HPC SDK (nvfortran compiler)")
endif()

message(STATUS "====================================")
