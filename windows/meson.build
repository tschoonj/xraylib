fs = import('fs')

windows_src_dir = meson.current_source_dir()
windows_bld_dir = meson.current_build_dir()

unix2dos_cmd = find_program('unix2dos', required : true)
lib_cmd = find_program('lib', required : true)
iscc_cmd = find_program('iscc', required: true)

unix2dos_input_files = {
    join_paths(project_source_root, 'README') : 'README.txt',
    join_paths(project_source_root, 'AUTHORS') :  'AUTHORS.txt',
    join_paths(project_source_root, 'TODO') : 'TODO.txt',
    join_paths(project_source_root, 'Changelog') :  'Changelog.txt',
    join_paths(windows_src_dir, 'README') : 'README2.txt',
}

foreach f: xraylib_headers + doc_files + windows_installer_example_files + pascal_source_files + cplusplus_source_files
    unix2dos_input_files += {
        f.full_path() : fs.name(f)
    }
endforeach

unix2dos_custom_targets = []

foreach input, output : unix2dos_input_files
    unix2dos_custom_targets += custom_target(
        command: [unix2dos_cmd, '-n', '@INPUT@', '@OUTPUT@'],
        input: [input],
        output: [output],
        build_by_default: true
    )
endforeach

lib_custom_target = custom_target(
    command: [
        lib_cmd,
        '/machine:X64',
        '/def:@0@'.format(libxrl_def_file),
        '/out:@OUTPUT0@',
    ],
    output: [
        'libxrl-@0@.lib'.format(lib_current_minus_age),
        'libxrl-@0@.exp'.format(lib_current_minus_age),
    ],
    build_by_default: true,
    depends: [xraylib_lib] 
)

iscc_custom_target = custom_target(
    command: [
        iscc_cmd,
        '-dXRL64',
        '-dLIB_CURRENT_MINUS_AGE=@0@'.format(lib_current_minus_age),
        '-dMyAppVersion=@0@'.format(meson.project_version()),
        '-dabs_top_srcdir_win=@0@'.format(project_source_root),
        '-dabs_top_builddir_win=@0@'.format(project_build_root),
        '@INPUT0@',
    ],
    input: files('xraylib.iss'),
    output: [
        'xraylib-@0@-win64.exe'.format(meson.project_version())
    ],
    build_by_default: true,
    depends: [lib_custom_target, unix2dos_custom_targets]
)
