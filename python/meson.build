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
            '-py3',
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
        install_tag: ['python-runtime']
    )
    python_enabled = true
endif

# meson has builtin support for cython as a language but it doesn't support cython3 unless you use a very recent version
cython = find_program('cython', 'cython3', 'cython-3', 'cython' + python.language_version(), 'cython-' + python.language_version(), required : get_option('python-numpy-bindings'))
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
            'language_level=3,boundscheck=False,wraparound=False,cdivision=True',
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
        install_tag: ['python-runtime'],
    )
    python_numpy_enabled = true
endif

subdir('tests')
