cplusplus_source_files = files('xraylib++.h')

install_headers(cplusplus_source_files, subdir: 'xraylib')

cplusplus_source_dir = meson.current_source_dir()

cplusplus_enabled = false

if add_languages('cpp', required: false, native: false)
    subdir('tests')
    cplusplus_enabled = true
else
    warning('The C++ tests will not get built and run')
endif