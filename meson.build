project('xraylib',
    'c',
    meson_version: '>= 0.60.0',
    version: '4.2.0',
    license: 'BSD',
    default_options: ['cpp_std=c++11',]
)


# versioning stuff -> maintain compatibility with libtool!
# # a) If binary compatibility has been broken (eg removed or changed interfaces)
# #    change to C+1:0:0.
# # b) If interfaces have been changed or added, but binary compatibility has
# #    been preserved, change to C+1:0:A+1
# # c) If the interface is the same as the previous version, change to C:R+1:A
lib_current = 12
lib_revision = 1
lib_age = 1

lib_current_minus_age = lib_current - lib_age

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
]

if host_system != 'windows'
    funcs += ['strndup']
endif

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
    if cc.get_id() == 'msvc' or cc.get_id() == 'clang-cl'
      config_h_data.set('XRL_EXTERN', '__declspec(dllexport) extern')
    elif cc.has_argument('-fvisibility=hidden')
      config_h_data.set('XRL_EXTERN', '__attribute__((visibility("default"))) __declspec(dllexport) extern')
    endif
  elif cc.has_argument('-fvisibility=hidden')
    config_h_data.set('XRL_EXTERN', '__attribute__((visibility("default"))) extern')
  endif
else
  config_h_data.set('XRL_EXTERN', 'extern')
endif
 
if cc.get_id() not in ['msvc', 'clang-cl'] and cc.has_header('complex.h')
  config_h_data.set('HAVE_COMPLEX_H', true)
endif

configure_file(output : 'config.h', configuration : config_h_data)

m_dep = cc.find_library('m', required : false)
xraylib_build_dep = [m_dep]

pkgconfig = import('pkgconfig')

subdir('include')
subdir('doc')
subdir('src')
subdir('tests')
subdir('cplusplus')
subdir('pascal')

python_enabled = false
python_numpy_enabled = false

if not (get_option('python-bindings').disabled() and get_option('python-numpy-bindings').disabled())
  pymod = import('python')
  python = pymod.find_installation(required: false, modules: ['numpy',])

  if not python.found() or python.language_version().version_compare('< 3.6')
    if get_option('python-bindings').enabled() or get_option('python-numpy-bindings').enabled()
      error('No suitable Python installation was found. The Python bindings cannot not be built')
    else
      warning('No suitable Python installation was found. The Python bindings cannot not be built')
    endif
  else
    python_dep = python.dependency()
    subdir('python')
  endif
endif

fortran_enabled = false
if add_languages('fortran', required: get_option('fortran-bindings'), native: false)
  fc = meson.get_compiler('fortran')

  c_loc_test = '''
MODULE f2003_test
USE, INTRINSIC ::ISO_C_BINDING

TYPE, BIND(C) :: test_C
	INTEGER (C_INT) :: arrayLen
	TYPE (C_PTR) :: cpointer
ENDTYPE

TYPE :: test_F
	INTEGER (C_INT), DIMENSION(:), POINTER :: fpointer
ENDTYPE

INTERFACE
	FUNCTION foo (bar,morebar) BIND(C,NAME='foo')
		USE, INTRINSIC ::ISO_C_BINDING
		IMPLICIT NONE
		REAL (KIND=C_DOUBLE) :: foo
		INTEGER (KIND=C_INT), INTENT(IN) :: bar
		REAL (KIND=C_DOUBLE), INTENT(IN) :: morebar
	ENDFUNCTION foo
	FUNCTION strlen(s) BIND(C,NAME='strlen')
		USE, INTRINSIC ::ISO_C_BINDING
		IMPLICIT NONE
		CHARACTER (KIND=C_CHAR),DIMENSION(*) :: s
		INTEGER (C_SIZE_T) :: strlen
	ENDFUNCTION
ENDINTERFACE
ENDMODULE f2003_test
PROGRAM f2003_main
USE f2003_test
IMPLICIT NONE

CHARACTER (LEN=10,KIND=C_CHAR) :: string = C_CHAR_'123456789' // C_NULL_CHAR
TYPE (test_C) :: tester_C
TYPE (test_F) :: tester_F



IF (strlen(string) /= 9) THEN
	CALL EXIT(1)
ELSE
	CALL EXIT(0)
ENDIF

!next line should produce a compile-time error when using g95
CALL C_F_POINTER(tester_C%cpointer,tester_F%fpointer,[tester_C%arrayLen])

!use c_loc
tester_C%cpointer = C_LOC(tester_F%fpointer(1))


ENDPROGRAM f2003_main
'''

  if fc.compiles(c_loc_test)
    subdir('fortran')
  else
    if get_option('fortran-bindings').enabled()
      error('The Fortran compiler does not support the required 2003 standard features')
    else
      warning('The Fortran compiler does not support the required 2003 standard features')
    endif
  endif
endif

subdir('example')

if get_option('windows-installer').enabled()
  if host_system == 'windows' and \
      cc.get_id() == 'gcc' and \
      meson.version().version_compare('>=1.4.0')
    subdir('windows')
  else
    warning('The windows-installer option is only supported on Windows systems with gcc selected as compiler')
  endif
endif
