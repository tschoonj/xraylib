# get numpy header location
numpy_include_script = '''
import numpy
print(numpy.get_include())
'''

rv = run_command(python, '-c', numpy_include_script, check: false)
if rv.returncode() != 0
  error('Could not retrieve numpy include location. Please check that numpy has been installed.')
endif
numpy_header_location = rv.stdout().strip()

# free-threading support (PEP 703) is only meaningful when actually targeting a
# free-threaded (no-GIL) build of Python, e.g. 3.13t/3.14t
python_freethreaded_script = '''
import sysconfig
print(1 if sysconfig.get_config_var("Py_GIL_DISABLED") else 0)
'''
python_freethreaded_rv = run_command(python, '-c', python_freethreaded_script, check: false)
python_is_freethreaded = python_freethreaded_rv.returncode() == 0 and python_freethreaded_rv.stdout().strip() == '1'


python_error_flags = [
  '-Wno-error=cpp',
  '-Wno-error=attributes',
  '-Wno-error=deprecated-declarations',
  '-Wno-error=unreachable-code',
  '-Wno-error=ignored-optimization-argument',
]

python_error_flags = cc.get_supported_arguments(python_error_flags)

# We cannot set this until cython 3 is adopted everywhere
# numpy_cflags = ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']

pydir = meson.current_build_dir()

swig = find_program(get_option('swig'), 'swig', required : get_option('python-bindings'))

# -nogil (free-threading support) was only added to swig in version 4.4.0 (see
# https://github.com/swig/swig/issues/3215); only request it when actually
# targeting a free-threaded Python build and swig is new enough.
# (-py3 is a legacy, unrelated flag: it has been deprecated and a no-op since
# swig 4.1.0, so it is simply omitted rather than kept as a fallback)
swig_extra_args = []
if swig.found() and python_is_freethreaded
    swig_version_rv = run_command(swig, '-version', check: false)
    swig_version_output = swig_version_rv.stdout().strip()
    if swig_version_output == ''
        swig_version_output = swig_version_rv.stderr().strip()
    endif
    swig_version = ''
    foreach line : swig_version_output.split('\n')
        stripped_line = line.strip()
        if swig_version == '' and stripped_line.startswith('SWIG Version')
            parts = stripped_line.split(' ')
            swig_version = parts[parts.length() - 1]
        endif
    endforeach
    if swig_version != '' and swig_version.version_compare('>=4.4.0')
        swig_extra_args = ['-nogil']
    else
        warning('swig >= 4.4.0 is required to build free-threading compatible Python bindings; the GIL will be re-enabled when the module is imported')
    endif
endif

if swig.found()
    xraylib_wrap_c = custom_target('xraylib_wrap_c',
        output : ['xraylib_wrap.c', 'xraylib.py'],
        input : swig_interface,
        depend_files : xraylib_headers,
        command : [
            swig,
            '-DVERSION=\'@0@\''.format(meson.project_version()),
            '-includeall',
            '-I@0@'.format(meson.current_source_dir()),
            '-I@0@'.format(include_source_dir),
            '-I@0@'.format(src_source_dir),
            '-o', '@OUTPUT0@',
            '-python',
            ] + swig_extra_args + [
            '@INPUT@',
            ],
        install_dir: [false, python.get_install_dir(pure: false)],
        install: true,
        install_tag: [false, 'python-runtime']

    )
    xraylib_ext = python.extension_module('_xraylib', xraylib_wrap_c[0],
        dependencies : [python_dep, xraylib_lib_dep],
        include_directories: extra_include_dirs,
        install: true,
        install_dir: python.get_install_dir(pure: false),
        c_args: core_c_args + ['-I' + numpy_header_location] + python_error_flags,
        install_tag: 'python-runtime'
    )
    python_enabled = true
endif

# meson has builtin support for cython as a language but it doesn't support cython3 unless you use a very recent version
cython = find_program('cython', 'cython3', 'cython-3', 'cython' + python.language_version(), 'cython-' + python.language_version(), required : get_option('python-numpy-bindings'))

# the freethreading_compatible directive (free-threading support) was only added in
# Cython 3.1.0; only request it when actually targeting a free-threaded Python
# build and cython is new enough, to avoid an 'unknown compiler directive' error
cython_directives = 'language_level=3,boundscheck=False,wraparound=False,cdivision=True'
if cython.found() and python_is_freethreaded
    cython_version_rv = run_command(cython, '--version', check: false)
    cython_version_output = cython_version_rv.stdout().strip()
    if cython_version_output == ''
        cython_version_output = cython_version_rv.stderr().strip()
    endif
    cython_version = ''
    foreach line : cython_version_output.split('\n')
        stripped_line = line.strip()
        if cython_version == '' and stripped_line.to_lower().startswith('cython version')
            parts = stripped_line.split(' ')
            cython_version = parts[parts.length() - 1]
        endif
    endforeach
    if cython_version != '' and cython_version.version_compare('>=3.1.0')
        cython_directives += ',freethreading_compatible=True'
    else
        warning('Cython >= 3.1.0 is required to build free-threading compatible Python-NumPy bindings; the GIL will be re-enabled when the module is imported')
    endif
endif

if cython.found()
    deps = [python_dep, xraylib_lib_dep]
    # lld-link doesnt find the openmp import libraries
    if cc.get_id() != 'clang-cl'
      deps += [dependency('openmp', language: 'c', required: false)]
    endif
    xraylib_np_c = custom_target('xraylib_np_c',
        output : 'xraylib_np.c',
        input : 'xraylib_np.pyx',
        depend_files : 'xraylib_np_c.pxd',
        command : [
            cython,
            '-X',
            cython_directives,
            '@INPUT@',
            '-o',
            '@OUTPUT@',
        ],
    )
    xraylib_np_ext = python.extension_module('xraylib_np', xraylib_np_c,
        dependencies : deps,
        include_directories: extra_include_dirs,
        install: true,
        install_dir: python.get_install_dir(pure: false),
        c_args: core_c_args + ['-I' + numpy_header_location] + python_error_flags,
        install_tag: 'python-runtime',
    )
    python_numpy_enabled = true
endif

subdir('tests')
