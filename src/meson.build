src_source_dir = meson.current_source_dir()
src_build_dir = meson.current_build_dir()

core_c_args = [
  '-DHAVE_CONFIG_H',
  '-D_GNU_SOURCE',
]

xraylib_error_flags = [
  '-Werror=implicit-function-declaration',
  '-Werror=implicit',
  '-Werror=unused-function',
  '-Werror=parentheses',
  '-Werror=unused-result',
]

# clang-cl really doesnt like posix functions like strerror etc
if cc.get_id() != 'clang-cl'
    xraylib_error_flags += [
        '-Werror=deprecated',
        '-Werror=deprecated-declarations',
    ]
endif

xraylib_error_flags = cc.get_supported_arguments(xraylib_error_flags)

extra_include_dirs = include_directories('..', '.', '../include',)

shared_sources = files(
    'atomicweight.c',
    'auger_trans.c',
    'coskron.c',
    'cross_sections.c',
    'crystal_diffraction.c',
    'fi.c',
    'fii.c',
    'fluor_yield.c',
    'radrate.c',
    'scattering.c',
    'splint.c',
    'splint.h',
    'xraylib-aux.c',
    'xraylib-error.c',
    'xraylib-error-private.h',
    'xrayglob.h',
    'xrayvars.c',
    'xrayvars.h',
)

libprdata_sources = shared_sources + files(
    'xrayglob.c',
    'xrayfiles.c',
    'xrf_cross_sections_aux-private.h',
    'xrf_cross_sections_aux-private.c',
)

prdata_sources = files(
    'pr_data.c'
)

prdata_lib = static_library(
  'prdata',
  libprdata_sources,
  dependencies: xraylib_build_dep,
  c_args: core_c_args + xraylib_error_flags,
  include_directories: extra_include_dirs,
  native: true,
)

prdata_lib_dep = declare_dependency(
  link_with: prdata_lib,
  include_directories: extra_include_dirs,
  dependencies: xraylib_build_dep,
)

prdata_exec = executable(
  'prdata',
  prdata_sources,
  dependencies: prdata_lib_dep,
  c_args: core_c_args + xraylib_error_flags,
  native: true,
)

xrayglob_inline = custom_target(
    'xrayglob_inline.c',
    output: ['xrayglob_inline.c'],
    command: [prdata_exec, project_source_root, '@OUTPUT@']
)

libxrl_sources = shared_sources + [xrayglob_inline] + files(
    'atomiclevelwidth.c',
    'comptonprofiles.c',
    'cs_barns.c',
    'cs_cp.c',
    'cs_line.c',
    'densities.c',
    'edges.c',
    'fluor_lines.c',
    'jump.c',
    'kissel_pe.c',
    'polarized.c',
    'refractive_indices.c',
    'xrayfiles_inline.c',
    'xraylib-deprecated-private.h',
    'xraylib-nist-compounds.c',
    'xraylib-nist-compounds-internal.h',
    'xraylib-parser.c',
    'xraylib-radionuclides-internal.h',
    'xraylib-radionuclides.c',
    'xrf_cross_sections_aux.h',
    'xrf_cross_sections_aux.c',
)

libxrl_def_file = join_paths(src_build_dir, 'libxrl-@0@.def'.format(lib_current_minus_age))

libxrl_link_args = cc.get_supported_link_arguments(['-Wl,--output-def,@0@'.format(libxrl_def_file)])

xraylib_lib = library(
  'xrl',
  libxrl_sources,
  version: version,
  darwin_versions: darwin_versions,
  dependencies: xraylib_build_dep,
  install: true,
  c_args: core_c_args + xraylib_error_flags,
  gnu_symbol_visibility: 'hidden',
  include_directories: extra_include_dirs,
  link_args: libxrl_link_args
)

xraylib_lib_dep = declare_dependency(
  link_with: xraylib_lib,
  include_directories: extra_include_dirs,
  dependencies: xraylib_build_dep,
)

swig_interface = files('xraylib.i')

pkgconfig.generate(
    xraylib_lib,
    name: 'xraylib',
    description: 'xraylib provides access to x-ray related physical data',
    filebase: 'libxrl',
    subdirs: ['xraylib'],
)
