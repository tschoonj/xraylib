xraylib_fortran_lib = library(
    'xrlf03',
    files('xraylib_wrap.F90'),
    version: version,
    darwin_versions: darwin_versions,
    dependencies: [xraylib_lib_dep],
    install: true,
    include_directories: include_directories('.'),
    link_language: 'fortran',
)

xraylib_fortran_lib_dep = declare_dependency(link_with: xraylib_fortran_lib, include_directories: ['.'])

pkgconfig.generate(
    xraylib_fortran_lib,
    name: 'xraylib Fortran 2003 bindings',
    description: 'xraylib provides access to x-ray related physical data',
    filebase: 'libxrlf03',
    subdirs: ['xraylib'],
)

# need to setup post install script for mod file

subdir('tests')
