# This Python file uses the following encoding: utf-8
#!/usr/bin/env python3

import sys
if sys.version_info.minor < 6:
        raise Exception("Execute this script with at least python 3.6 to ensure all dicts are ordered!")

XRL_FUNCTIONS = {
#		'AtomicWeight': {'Z': int},
#		'ElementDensity': {'Z': int},
#		'CS_Total': {'Z': int, 'E': float},
#		'CS_Photo': {'Z': int, 'E': float},
#		'CS_Rayl': {'Z': int, 'E': float},
#		'CS_Compt': {'Z': int, 'E': float},
#		'CS_Energy': {'Z': int, 'E': float},
#		'CS_KN': {'E': float},
#		'CSb_Total': {'Z': int, 'E': float},
#		'CSb_Photo': {'Z': int, 'E': float},
#		'CSb_Rayl': {'Z': int, 'E': float},
#		'CSb_Compt': {'Z': int, 'E': float},
#		'DCS_Thoms': {'theta': float},
#		'DCS_KN': {'E':float, 'theta': float},
#		'DCS_Rayl': {'Z': int, 'E':float, 'theta': float},
#		'DCS_Compt': {'Z': int, 'E':float, 'theta': float},
#		'DCSb_Rayl': {'Z': int, 'E':float, 'theta': float},
#		'DCSb_Compt': {'Z': int, 'E':float, 'theta': float},
#		'DCSP_Thoms': {'theta': float, 'phi': float},
#		'DCSP_KN': {'E': float, 'theta': float, 'phi': float},
#		'DCSP_Rayl': {'Z': int, 'E': float, 'theta': float, 'phi': float},
#		'DCSP_Compt': {'Z': int, 'E': float, 'theta': float, 'phi': float},
#		'DCSPb_Rayl': {'Z': int, 'E': float, 'theta': float, 'phi': float},
#		'DCSPb_Compt': {'Z': int, 'E': float, 'theta': float, 'phi': float},
#		'FF_Rayl': {'Z': int, 'q': float},
#		'SF_Compt': {'Z': int, 'q': float},
#		'MomentTransf': {'E': float, 'theta': float},
                'LineEnergy': {'returnType': float, 'args': {'Z': int, 'line': int}},
                'FluorYield': {'returnType': float, 'args': {'Z': int, 'shell': int}},
#		'CosKronTransProb': {'Z': int, 'trans': int},
#		'EdgeEnergy': {'Z': int, 'shell': int},
#		'JumpFactor': {'Z': int, 'shell': int},
#		'CS_FluorLine': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine': {'Z': int, 'line': int, 'E': float},
#		'RadRate': {'Z': int, 'line': int},
#		'ComptonEnergy': {'E0': float, 'theta': float},
#		'Fi': {'Z': int, 'E': float},
#		'Fii': {'Z': int, 'E': float},
#		'CS_Photo_Total': {'Z': int, 'E': float},
#		'CSb_Photo_Total': {'Z': int, 'E': float},
#		'CS_Photo_Partial': {'Z': int, 'shell': int, 'E': float},
#		'CSb_Photo_Partial': {'Z': int, 'shell': int, 'E': float},
#		'CS_Total_Kissel': {'Z': int, 'E': float},
#		'CSb_Total_Kissel': {'Z': int, 'E': float},
#		'ComptonProfile': {'Z': int, 'pz': float},
#		'ComptonProfile_Partial': {'Z': int, 'shell': int, 'pz': float},
#		'ElectronConfig': {'Z': int, 'shell': int},
#		'ElectronConfig_Biggs': {'Z': int, 'shell': int},
#		'AtomicLevelWidth': {'Z': int, 'shell': int},
#		'AugerRate': {'Z': int, 'auger_trans': int},
#		'AugerYield': {'Z': int, 'shell': int},
#		'CS_FluorLine_Kissel': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine_Kissel': {'Z': int, 'line': int, 'E': float},
#		'CS_FluorLine_Kissel_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine_Kissel_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CS_FluorLine_Kissel_Nonradiative_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine_Kissel_Nonradiative_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CS_FluorLine_Kissel_Radiative_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine_Kissel_Radiative_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CS_FluorLine_Kissel_no_Cascade': {'Z': int, 'line': int, 'E': float},
#		'CSb_FluorLine_Kissel_no_Cascade': {'Z': int, 'line': int, 'E': float},
#		'PL1_pure_kissel': {'Z': int, 'E': float},
#		'PL1_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
#		'PL1_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
#		'PL1_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float},
#		'PL2_pure_kissel': {'Z': int, 'E': float, 'PL1': float},
#		'PL2_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
#		'PL2_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
#		'PL2_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float},
#		'PL3_pure_kissel': {'Z': int, 'E': float, 'PL1': float, 'PL2': float},
#		'PL3_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
#		'PL3_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
#		'PL3_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float},
#		'PM1_pure_kissel': {'Z': int, 'E': float},
#		'PM1_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
#		'PM1_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
#		'PM1_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float},
#		'PM2_pure_kissel': {'Z': int, 'E': float, 'PM1': float},
#		'PM2_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
#		'PM2_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
#		'PM2_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float},
#		'PM3_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float},
#		'PM3_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
#		'PM3_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
#		'PM3_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float},
#		'PM4_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float, 'PM3': float},
#		'PM4_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
#		'PM4_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
#		'PM4_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float},
#		'PM5_pure_kissel': {'Z': int, 'E': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
#		'PM5_rad_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
#		'PM5_auger_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},
#		'PM5_full_cascade_kissel': {'Z': int, 'E': float, 'PK': float, 'PL1': float, 'PL2': float, 'PL3': float, 'PM1': float, 'PM2': float, 'PM3': float, 'PM4': float},

                'CS_Total_CP': {'returnType': float, 'args': {'compound': str, 'E': float}},
#		'CS_Photo_CP': {'compound': str, 'E': float},
#		'CS_Rayl_CP': {'compound': str, 'E': float},
#		'CS_Compt_CP': {'compound': str, 'E': float},
#		'CS_Energy_CP': {'compound': str, 'E': float},
#		'CSb_Total_CP': {'compound': str, 'E': float},
#		'CSb_Photo_CP': {'compound': str, 'E': float},
#		'CSb_Rayl_CP': {'compound': str, 'E': float},
#		'CSb_Compt_CP': {'compound': str, 'E': float},
#		'DCS_Rayl_CP': {'compound': str, 'E': float, 'theta': float},
#		'DCS_Compt_CP': {'compound': str, 'E': float, 'theta': float},
#		'DCSb_Rayl_CP': {'compound': str, 'E': float, 'theta': float},
#		'DCSb_Compt_CP': {'compound': str, 'E': float, 'theta': float},
#		'DCSP_Rayl_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'DCSP_Compt_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'DCSP_Rayl_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'DCSP_Compt_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'DCSPb_Rayl_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'DCSPb_Compt_CP': {'compound': str, 'E': float, 'theta': float, 'phi': float},
#		'CS_Photo_Total_CP': {'compound': str, 'E': float},
#		'CS_Total_Kissel_CP': {'compound': str, 'E': float},
#		'CSb_Photo_Total_CP': {'compound': str, 'E': float},
#		'CSb_Total_Kissel_CP': {'compound': str, 'E': float},
#                'Refractive_Index_Re': {'compound': str, 'E': float, 'density': float},
#                'Refractive_Index_Im': {'compound': str, 'E': float, 'density': float}
   }

def generate_func_args_for_str(arg_name: str) -> str:
    return f"std::string {arg_name}"


def generate_func_args_for_float(arg_name: str) -> str:
    return f"double {arg_name}"


def generate_func_args_for_int(arg_name: str) -> str:
    return f"int {arg_name}"


GENERATE_FUNC_ARGS = {
        str: generate_func_args_for_str,
        float: generate_func_args_for_float,
        int: generate_func_args_for_int
}

def GENERATE_FUNC_BODY(funcName, args):

    argList = ", ".join([f'{key}' for key in args.keys()])
    argList = argList.replace("compound,", "compound.data(),")  #replace std::string call with call to underlying C chars
    string = "\n    xrl_error *error = nullptr;\n"
    string += "    double rv = ::{}({}, &error);\n".format(funcName, argList)
    string += "    process_error(error);\n    return rv;\n}"
    return string



def replace_function_calls(iface_f) -> str:
    for key in XRL_FUNCTIONS.keys():
        returnType = ""
        for funcKey in XRL_FUNCTIONS[key]:
            if funcKey == 'returnType':
               returnType = "\n\n" + GENERATE_FUNC_ARGS[XRL_FUNCTIONS[key][funcKey]]("")
            if funcKey == 'args':
                args = XRL_FUNCTIONS[key][funcKey]
                funcHeader = "{}({})".format(key, ", ".join([GENERATE_FUNC_ARGS[arg_type](arg_name) for arg_name, arg_type in args.items()]))
                funcBody = GENERATE_FUNC_BODY(key, args)

        iface_f.write(returnType + funcHeader + " {" + funcBody)


def compoundParser(iface_f) -> str:
    string = r'''
std::unique_ptr<compoundData> CompoundParser(const std::string &compoundString)
{
    xrl_error *error = nullptr;
    auto compound = std::unique_ptr<compoundData>(CompoundParser(compoundString.data(), &error));
    process_error(error);
    return compound;
}

//std::unique_ptr<


'''
    iface_f.write(string)


def write_header(iface_f) -> str:
    iface = r'''
#ifndef XRAYLIB_WRAP_H
#define XRAYLIB_WRAP_H

#include <xraylib.h>

// headers containing std c++ exception defintions
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

namespace xrlcpp {

/* process_error translates error codes from xraylib to c++
* and throws the corresponding std exception.
* c++ reference: https://en.cppreference.com/w/cpp/error/exception
* and https://stackoverflow.com/questions/11938979/what-exception-classes-are-in-the-standard-c-library
*/
void process_error(xrl_error *error) {

    if (!error)
        return;

    switch (error->code) {

    case XRL_ERROR_MEMORY: /* set in case of a memory allocation problem */
        throw std::bad_alloc();
    case XRL_ERROR_INVALID_ARGUMENT: /* set in case an invalid argument gets passed to a routine */
        throw std::invalid_argument(error->message);
    case XRL_ERROR_IO: /* set in case an error involving input/output occurred */
        throw std::ios_base::failure(error->message);
    case XRL_ERROR_TYPE: /* set in case an error involving type conversion occurred */
        throw std::bad_cast();
    case XRL_ERROR_UNSUPPORTED: /* set in case an unsupported feature has been requested */
        throw std::invalid_argument(error->message);
    case XRL_ERROR_RUNTIME:
        throw std::runtime_error(error->message); /* set in case an unexpected runtime error occurred */
    }
}
'''
    iface_f.write(iface)


def write_footer(iface_f) -> str:
    iface = r'''

}  // end namespace xrlcpp

#endif // XRAYLIB_WRAP_H
'''
    iface_f.write(iface)


def main():
    iface_f = open('xraylib_test.h', 'w')
    iface_f.write('/* This file has been generated automatically using generate-code.py */\n')
    write_header(iface_f)
    iface_f.write('\n\n/* XRL FUNCTIONS */')
    replace_function_calls(iface_f)
    iface_f.write('\n\n/* END OF XRL FUNCTIONS */\n')
    iface_f.write('\n\n/* COMPOUND PARSER */')
    compoundParser(iface_f)
    write_footer(iface_f)

if __name__ == '__main__':
    main()
