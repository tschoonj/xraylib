/* This file has been generated automatically using generate-code.py */

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
void process_error(xrl_error *error, std::string funcName) {

    if (!error)
        return;

    switch (error->code) {

    case XRL_ERROR_MEMORY: /* set in case of a memory allocation problem */
        throw std::bad_alloc();
    case XRL_ERROR_INVALID_ARGUMENT: /* set in case an invalid argument gets passed to a routine */
        throw std::invalid_argument(funcName + error->message);
    case XRL_ERROR_IO: /* set in case an error involving input/output occurred */
        throw std::ios_base::failure(error->message);
    case XRL_ERROR_TYPE: /* set in case an error involving type conversion occurred */
        throw std::bad_cast();
    case XRL_ERROR_UNSUPPORTED: /* set in case an unsupported feature has been requested */
        throw std::invalid_argument(funcName + error->message);
    case XRL_ERROR_RUNTIME:
        throw std::runtime_error(error->message); /* set in case an unexpected runtime error occurred */
    }
}


/* XRL FUNCTIONS */

double LineEnergy(int Z, int line) {
    xrl_error *error = nullptr;
    double rv = ::LineEnergy(Z, line, &error);
    process_error(error, "");
    return rv;
}

double FluorYield(int Z, int shell) {
    xrl_error *error = nullptr;
    double rv = ::FluorYield(Z, shell, &error);
    process_error(error, "");
    return rv;
}

double CS_Total_CP(std::string compound, double E) {
    xrl_error *error = nullptr;
    double rv = ::CS_Total_CP(compound.data(), E, &error);
    process_error(error, "CS_Total_CP");
    return rv;
}

/* END OF XRL FUNCTIONS */


/* COMPOUND PARSER */
std::unique_ptr<compoundData> CompoundParser(const std::string &compoundString)
{
    xrl_error *error = nullptr;
    auto compound = std::unique_ptr<compoundData>(CompoundParser(compoundString.data(), &error));
    process_error(error, "");
    return compound;
}

//std::unique_ptr<




}  // end namespace xrlcpp

#endif // XRAYLIB_WRAP_H
