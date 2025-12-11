cmake_minimum_required(VERSION 3.12)
project(PHYEX_ICE_ADJUST Fortran)

# Set Fortran compiler flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O2 -fPIC -cpp")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fbacktrace -fcheck=all")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native")

# Set module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

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
# TURBULENCE MODULES (modd_* from turb/)
# ============================================================================
set(TURB_MODD_SOURCES
    ${PHYEX_DIR}/turb/modd_cturb.F90
    ${PHYEX_DIR}/turb/modd_turbn.F90
    ${PHYEX_DIR}/turb/modd_param_mfshalln.F90
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
)

# ============================================================================
# MAIN MICROPHYSICS ROUTINES (ice_adjust, rain_ice, and dependencies)
# ============================================================================
set(MAIN_SOURCES
    ${PHYEX_DIR}/micro/condensation.F90
    ${PHYEX_DIR}/micro/ice_adjust.F90
    ${PHYEX_DIR}/micro/rain_ice.F90
)

# ============================================================================
# COMBINE ALL SOURCES IN PROPER ORDER
# ============================================================================
set(ALL_SOURCES
    # Level 0: External dependency stubs (must be first!)
    ${STUB_SOURCES}
    
    # Level 1: Basic modules without dependencies
    ${AUX_MODD_SOURCES}
    
    # Level 2: Microphysics and turbulence data modules
    ${MICRO_MODD_SOURCES}
    ${TURB_MODD_SOURCES}
    
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
    
    # Level 8: Microphysics functions
    ${MICRO_FUNC_SOURCES}
    
    # Level 9: Main routines
    ${MAIN_SOURCES}
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

# Installation rules for scikit-build-core
install(TARGETS ice_adjust_phyex
    LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
    RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/ice3
)

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
message(STATUS "Compiler Flags: ${CMAKE_Fortran_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PHYEX Directory: ${PHYEX_DIR}")
message(STATUS "Module Directory: ${CMAKE_Fortran_MODULE_DIRECTORY}")
message(STATUS "Total source files: ${CMAKE_MATCH_COUNT}")
message(STATUS "====================================")
