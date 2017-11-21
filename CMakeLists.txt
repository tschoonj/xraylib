cmake_minimum_required (VERSION 2.8.12)

add_library(xraylib
    src/atomiclevelwidth.c
    src/atomicweight.c
    src/auger_trans.c
    src/comptonprofiles.c
    src/coskron.c
    src/cross_sections.c
    src/crystal_diffraction.c
    src/cs_barns.c
    src/cs_cp.c
    src/cs_line.c
    src/densities.c
    src/edges.c
    src/fi.c
    src/fii.c
    src/fluor_lines.c
    src/fluor_yield.c
    src/jump.c
    src/kissel_pe.c
    src/polarized.c
    src/pr_data.c
    src/radrate.c
    src/refractive_indices.c
    src/scattering.c
    src/splint.c
    src/xrayfiles.c
    src/xrayfiles_inline.c
    src/xrayglob.c
    src/xraylib-aux.c
    src/xraylib-nist-compounds.c
    src/xraylib-parser.c
    src/xraylib-radionuclides.c
    src/xrayvars.c
    src/xrf_cross_sections_aux.c
    src/splint.h
    src/xrayglob.h
    src/xraylib-aux.h
    src/xraylib-nist-compounds-internal.h
    src/xraylib-radionuclides-internal.h
    src/xrayvars.h
    src/xrf_cross_sections_aux.h
    include/lines_old.h
    include/xraylib-auger.h
    include/xraylib-crystal-diffraction.h
    include/xraylib-defs.h
    include/xraylib-lines.h
    include/xraylib-nist-compounds.h
    include/xraylib-parser.h
    include/xraylib-radionuclides.h
    include/xraylib-shells.h
    include/xraylib.h
    )

include(CheckFunctionExists)
check_function_exists(strdup HAVE_STRDUP)
check_function_exists(strndup HAVE_STRNDUP)
configure_file (${CMAKE_SOURCE_DIR}/src/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

target_include_directories(xraylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(xraylib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(xraylib PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(xrlexample6
    example/xrlexample6.cpp
    )
target_link_libraries(xrlexample6 xraylib)
