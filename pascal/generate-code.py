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

def generate_iface_arg_for_str(arg_name: str) -> str:
    return f"{arg_name}:string"

def generate_iface_arg_for_float(arg_name: str) -> str:
    return f"{arg_name}:double"

def generate_iface_arg_for_int(arg_name: str) -> str:
    return f"{arg_name}:longint"

GENERATE_IFACE_ARG = {
	str: generate_iface_arg_for_str,
	float: generate_iface_arg_for_float,
	int: generate_iface_arg_for_int
}

def generate_iface_c_arg_for_str(arg_name: str) -> str:
    return f"{arg_name}:PAnsiChar"

GENERATE_IFACE_C_ARG = {
	str: generate_iface_c_arg_for_str,
	float: generate_iface_arg_for_float,
	int: generate_iface_arg_for_int
}

def generate_additional_var_for_str(arg_name: str) -> str:
    return f"{arg_name}_c:PAnsiChar;\n"

GENERATE_ADDITIONAL_VAR = {
        str: generate_additional_var_for_str,        
	float: lambda arg_name: "",
	int: lambda arg_name: ""
}

def generate_preprocess_for_str(arg_name: str) -> str:
    return f"{arg_name}_c := PAnsichar(AnsiString({arg_name}));\n"

GENERATE_PREPROCESS = {
	str: generate_preprocess_for_str,
	float: lambda arg_name: "",
	int: lambda arg_name: ""
}

GENERATE_CALL = {
	str: lambda arg_name: f"{arg_name}_c",
	float: lambda arg_name: f"{arg_name}",
	int: lambda arg_name: f"{arg_name}"
}

def process_function(name: str, iface_f, impl_f) -> str:

    args = XRL_FUNCTIONS[name]
    iface = "function {}({}):double;\n".format(name, '; '.join([GENERATE_IFACE_ARG[arg_type](arg_name) for arg_name, arg_type in args.items()]))
    iface_f.write(iface)
    iface_c = "function {}_C({}; error:PPxrl_error):double;cdecl;external External_library name '{}';\n".format(name, '; '.join([GENERATE_IFACE_C_ARG[arg_type](arg_name) for arg_name, arg_type in args.items()]), name)
    impl_f.write(iface_c)
    additional_vars = ''.join([GENERATE_ADDITIONAL_VAR[arg_type](arg_name) for arg_name, arg_type in args.items()])
    preprocess = ''.join([GENERATE_PREPROCESS[arg_type](arg_name) for arg_name, arg_type in args.items()])
    arg_list =  ', '.join([GENERATE_CALL[arg_type](arg_name) for arg_name, arg_type in args.items()])
    impl = f'''
{iface}
var
    error: Pxrl_error;
    rv: double;
    {additional_vars}
begin
    error := nil;
    {preprocess}
    rv := {name}_C({arg_list}, @error);
    process_error(error);
    result := rv
end;

'''
    impl_f.write(impl)

with open('xraylib_iface.pas', 'w') as iface, open('xraylib_impl.pas', 'w') as impl:
    iface.write('{ This file has been generated automatically using generate-code.py}\n\n')
    impl.write('{ This file has been generated automatically using generate-code.py}\n\n')
    for function in XRL_FUNCTIONS.keys():
        process_function(function, iface, impl)
