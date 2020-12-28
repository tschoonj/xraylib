project('xraylib', ['c'], version:'4.0.0', license: 'BSD')

# versioning stuff -> maintain compatibility with libtool!
# # a) If binary compatibility has been broken (eg removed or changed interfaces)
# #    change to C+1:0:0.
# # b) If interfaces have been changed or added, but binary compatibility has
# #    been preserved, change to C+1:0:A+1
# # c) If the interface is the same as the previous version, change to C:R+1:A
lib_current = 12
lib_revision = 0
lib_age = 1

version = '@0@.@1@.@2@'.format((lib_current - lib_age), lib_age, lib_revision)

current = lib_current + 1
interface_age = lib_revision
darwin_versions = [current, '@0@.@1@'.format(current , interface_age)]

host_system = host_machine.system()

cc = meson.get_compiler('c')

# Use these instead of meson.source_root() and meson.build_root() in subdirectories.
# source_root() and build_root() are not useful, if this is a subproject.
project_source_root = meson.current_source_dir()
project_build_root = meson.current_build_dir()

# Configuration data
config_h_data = configuration_data()
config_h_data.set_quoted('PACKAGE_TARNAME', meson.project_name())
config_h_data.set_quoted('PACKAGE_VERSION', meson.project_version())
config_h_data.set_quoted('VERSION', meson.project_version())

funcs = [
  'strdup',
  '_strdup',
  'strndup',
]

foreach f : funcs
  if cc.has_function(f)
    define = 'HAVE_' + f.underscorify().to_upper()
    config_h_data.set(define, 1)
  endif
endforeach


special_funcs = []

if host_system == 'windows'
  special_funcs += ['_vscprintf', '_scprintf', '_vsnprintf', '_snprintf']
else
  special_funcs += ['vasprintf', 'asprintf']
endif

legacy_stdio_dep =  cc.find_library('legacy_stdio_definitions', required: false)

foreach f : special_funcs
  if not cc.has_function(f, args : '-D_GNU_SOURCE', dependencies: legacy_stdio_dep)
    error('Required function ' + f + ' not found on this platform')
  endif
endforeach

# Detect and set symbol visibility
if get_option('default_library') != 'static'
  if host_system == 'windows' or host_system == 'cygwin'
    config_h_data.set('WIN32', true)
    if cc.get_id() == 'msvc' or cc.get_id() == 'clang-cl'
      config_h_data.set('XRL_EXTERN', '__declspec(dllexport) extern')
    elif cc.has_argument('-fvisibility=hidden')
      config_h_data.set('XRL_EXTERN', '__attribute__((visibility("default"))) __declspec(dllexport) extern')
    endif
  elif cc.has_argument('-fvisibility=hidden')
    config_h_data.set('XRL_EXTERN', '__attribute__((visibility("default"))) extern')
  endif
endif
 
if cc.get_id() != 'msvc' and cc.has_header('complex.h')
  config_h_data.set('HAVE_COMPLEX_H', true)
endif

configure_file(output : 'config.h', configuration : config_h_data)

m_dep = cc.find_library('m', required : false)
xraylib_build_dep = [m_dep]

subdir('src')
subdir('tests')