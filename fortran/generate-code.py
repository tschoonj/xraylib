#!/usr/bin/env python3

import sys
if sys.version_info.minor < 6:
    raise Exception("Execute this script with at least python 3.6 to ensure all dicts are ordered!")

XRL_FUNCTIONS = {
    'AtomicWeight': {'Z': int},
    'ElementDensity': {'Z': int},
    'CS_Total': {'Z': int, 'E': float},
    'CS_Photo': {'Z': int, 'E': float},
    'CS_Rayl': {'Z': int, 'E': float},
    'CS_Compt': {'Z': int, 'E': float},
    'CS_Energy': {'Z': int, 'E': float},
    'CS_KN': {'E': float},
    'CSb_Total': {'Z': int, 'E': float},
    'CSb_Photo': {'Z': int, 'E': float},
    'CSb_Rayl': {'Z': int, 'E': float},
    'CSb_Compt': {'Z': int, 'E': float},
    'DCS_Thoms': {'theta': float},
    'DCS_KN': {'E':float, 'theta': float},
    'DCS_Rayl': {'Z': int, 'E':float, 'theta': float},
    'DCS_Compt': {'Z': int, 'E':float, 'theta': float},
    'DCSb_Rayl': {'Z': int, 'E':float, 'theta': float},
    'DCSb_Compt': {'Z': int, 'E':float, 'theta': float},
    'DCSP_Thoms': {'theta': float, 'phi': float},
    'DCSP_KN': {'E': float, 'theta': float, 'phi': float},
    'DCSP_Rayl': {'Z': int, 'E': float, 'theta': float, 'phi': float},
    'DCSP_Compt': {'Z': int, 'E': float, 'theta': float, 'phi': float},
    'DCSPb_Rayl': {'Z': int, 'E': float, 'theta': float, 'phi': float},
    'DCSPb_Compt': {'Z': int, 'E': float, 'theta': float, 'phi': float},
    'FF_Rayl': {'Z': int, 'q': float},
    'SF_Compt': {'Z': int, 'q': float},
    'MomentTransf': {'E': float, 'theta': float},
    'LineEnergy': {'Z': int, 'line': int},
    'FluorYield': {'Z': int, 'shell': int},
    'CosKronTransProb': {'Z': int, 'trans': int},
    'EdgeEnergy': {'Z': int, 'shell': int},
    'JumpFactor': {'Z': int, 'shell': int},
    'CS_FluorLine': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine': {'Z': int, 'line': int, 'E': float},
    'CS_FluorShell': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell': {'Z': int, 'shell': int, 'E': float},
    'RadRate': {'Z': int, 'line': int},
    'ComptonEnergy': {'E0': float, 'theta': float},
    'Fi': {'Z': int, 'E': float},
    'Fii': {'Z': int, 'E': float},
    'CS_Photo_Total': {'Z': int, 'E': float},
    'CSb_Photo_Total': {'Z': int, 'E': float},
    'CS_Photo_Partial': {'Z': int, 'shell': int, 'E': float},
    'CSb_Photo_Partial': {'Z': int, 'shell': int, 'E': float},
    'CS_Total_Kissel': {'Z': int, 'E': float},
    'CSb_Total_Kissel': {'Z': int, 'E': float},
    'ComptonProfile': {'Z': int, 'pz': float},
    'ComptonProfile_Partial': {'Z': int, 'shell': int, 'pz': float},
    'ElectronConfig': {'Z': int, 'shell': int},
    'ElectronConfig_Biggs': {'Z': int, 'shell': int},
    'AtomicLevelWidth': {'Z': int, 'shell': int},
    'AugerRate': {'Z': int, 'auger_trans': int},
    'AugerYield': {'Z': int, 'shell': int},
    'CS_FluorLine_Kissel': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine_Kissel': {'Z': int, 'line': int, 'E': float},
    'CS_FluorLine_Kissel_Cascade': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine_Kissel_Cascade': {'Z': int, 'line': int, 'E': float},
    'CS_FluorLine_Kissel_Nonradiative_Cascade': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine_Kissel_Nonradiative_Cascade': {'Z': int, 'line': int, 'E': float},
    'CS_FluorLine_Kissel_Radiative_Cascade': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine_Kissel_Radiative_Cascade': {'Z': int, 'line': int, 'E': float},
    'CS_FluorLine_Kissel_no_Cascade': {'Z': int, 'line': int, 'E': float},
    'CSb_FluorLine_Kissel_no_Cascade': {'Z': int, 'line': int, 'E': float},
    'CS_FluorShell_Kissel': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell_Kissel': {'Z': int, 'shell': int, 'E': float},
    'CS_FluorShell_Kissel_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell_Kissel_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CS_FluorShell_Kissel_Nonradiative_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell_Kissel_Nonradiative_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CS_FluorShell_Kissel_Radiative_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell_Kissel_Radiative_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CS_FluorShell_Kissel_no_Cascade': {'Z': int, 'shell': int, 'E': float},
    'CSb_FluorShell_Kissel_no_Cascade': {'Z': int, 'shell': int, 'E': float},
    'PL1_pure_kissel': {'Z': int, 'E': float},
    'PL1_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
    'PL1_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
    'PL1_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
    'PL2_pure_kissel': {'Z': int, 'E': float, 'PL1': float},
    'PL2_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
    'PL2_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
    'PL2_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
    'PL3_pure_kissel': {'Z': int, 'E': float, 'PL1': float, 'PL2': float},
    'PL3_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
    'PL3_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
    'PL3_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
    'PM1_pure_kissel': {'Z': int, 'E': float},
    'PM1_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
    'PM1_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
    'PM1_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
    'PM2_pure_kissel': {'Z': int, 'E': float, 'PM1': float},
    'PM2_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
    'PM2_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
    'PM2_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
    'PM3_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float},
    'PM3_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
    'PM3_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
    'PM3_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
    'PM4_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float, 'PM3': float},
    'PM4_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
    'PM4_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
    'PM4_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
    'PM5_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
    'PM5_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
    'PM5_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
    'PM5_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},

    'CS_Total_CP': {'compound': str, 'E': float},
    'CS_Photo_CP': {'compound': str, 'E': float},
    'CS_Rayl_CP': {'compound': str, 'E': float},
    'CS_Compt_CP': {'compound': str, 'E': float},
    'CS_Energy_CP': {'compound': str, 'E': float},
    'CSb_Total_CP': {'compound': str, 'E': float},
    'CSb_Photo_CP': {'compound': str, 'E': float},
    'CSb_Rayl_CP': {'compound': str, 'E': float},
    'CSb_Compt_CP': {'compound': str, 'E': float},
    'DCS_Rayl_CP': {'compound': str, 'E': float, 'theta': float},
    'DCS_Compt_CP': {'compound': str, 'E': float, 'theta': float},
    'DCSb_Rayl_CP': {'compound': str, 'E': float, 'theta': float},
    'DCSb_Compt_CP': {'compound': str, 'E': float, 'theta': float},
    'DCSP_Rayl_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
    'DCSP_Compt_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
    'DCSPb_Rayl_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
    'DCSPb_Compt_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
    'CS_Photo_Total_CP': {'compound': str, 'E': float},
    'CS_Total_Kissel_CP': {'compound': str, 'E': float},
    'CSb_Photo_Total_CP': {'compound': str, 'E': float},
    'CSb_Total_Kissel_CP': {'compound': str, 'E': float},
    'Refractive_Index_Re': {'compound': str, 'E': float, 'density': float},
    'Refractive_Index_Im': {'compound': str, 'E': float, 'density': float}
    }

def generate_declaration_for_str(arg_name: str) -> str:
    return f'''CHARACTER (KIND=C_CHAR,LEN=*), INTENT(IN) :: {arg_name} 
        CHARACTER (KIND=C_CHAR), DIMENSION(:), ALLOCATABLE, TARGET :: &
        {arg_name}_F'''

def generate_declaration_for_float(arg_name: str) -> str:
    return f"REAL (C_DOUBLE), INTENT(IN) :: {arg_name}"

def generate_declaration_for_int(arg_name: str) -> str:
    return f"INTEGER (C_INT), INTENT(IN) :: {arg_name}"

def generate_iface_declaration_for_str(arg_name: str) -> str:
    return f"TYPE (C_PTR), INTENT(IN), VALUE :: {arg_name}"

def generate_iface_declaration_for_float(arg_name: str) -> str:
    return f"REAL (C_DOUBLE), INTENT(IN), VALUE :: {arg_name}"

def generate_iface_declaration_for_int(arg_name: str) -> str:
    return f"INTEGER (C_INT), INTENT(IN), VALUE :: {arg_name}"




GENERATE_DECLARATION = {
    str: generate_declaration_for_str,
    float: generate_declaration_for_float,
    int: generate_declaration_for_int
}

GENERATE_IFACE_DECLARATION = {
    str: generate_iface_declaration_for_str,
    float: generate_iface_declaration_for_float,
    int: generate_iface_declaration_for_int
}

def generate_preprocess_for_str(arg_name: str) -> str:
    return f'''CALL stringF2C({arg_name}, {arg_name}_F)
'''

GENERATE_PREPROCESS = {
    str: generate_preprocess_for_str,
    float: lambda arg_name: "",
    int: lambda arg_name: ""
}

GENERATE_CALL = {
    str: lambda arg_name: f"C_LOC({arg_name}_F)",
    float: lambda arg_name: f"{arg_name}",
    int: lambda arg_name: f"{arg_name}"
}

def process_function(name: str) -> str:

    args = XRL_FUNCTIONS[name]

    arg_list = ', '.join(args.keys())

    arg_declarations = '\n        '.join([GENERATE_DECLARATION[arg_type](arg_name) for arg_name, arg_type in args.items()])

    arg_preprocess = ''.join([GENERATE_PREPROCESS[arg_type](arg_name) for arg_name, arg_type in args.items()])

    arg_calls =  ', '.join([GENERATE_CALL[arg_type](arg_name) for arg_name, arg_type in args.items()])
    arg_iface_declarations = '\n                '.join([GENERATE_IFACE_DECLARATION[arg_type](arg_name) for arg_name, arg_type in args.items()])

    return f'''
FUNCTION {name}({arg_list}, error) RESULT(rv)
        USE, INTRINSIC :: ISO_C_BINDING
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE

        {arg_declarations}
        REAL (C_DOUBLE) :: rv

        TYPE(xrl_error), POINTER, OPTIONAL :: error
        TYPE (C_PTR) :: errorPtr, errorPtrLoc
        TARGET :: errorPtr

        INTERFACE
            FUNCTION {name}C({arg_list}, error) &
            BIND(C,NAME='{name}')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                {arg_iface_declarations}
                REAL (C_DOUBLE) :: {name}C
                TYPE (C_PTR),INTENT(IN),VALUE :: error
            ENDFUNCTION {name}C
        ENDINTERFACE

        errorPtr = C_NULL_PTR
        errorPtrLoc = C_NULL_PTR

        IF (PRESENT(error)) THEN
                IF (.NOT. ASSOCIATED(error)) THEN
                        errorPtrLoc = C_LOC(errorPtr)
                ELSE
                        ! print warning
                        WRITE (error_unit, '(A)') & 
                        'error POINTER must be disassociated!'
                ENDIF
        ENDIF
        {arg_preprocess}
        rv = {name}C({arg_calls}, errorPtrLoc)

        IF (C_ASSOCIATED(errorPtr)) THEN
                CALL process_error(errorPtr, error)
        ENDIF
ENDFUNCTION {name}
'''


with open('xraylib_wrap_generated.F90', 'w') as f:
    f.write('! This file has been generated automatically using generate-code.py\n\n')
    f.write('\n'.join([process_function(function) for function in XRL_FUNCTIONS.keys()]))
