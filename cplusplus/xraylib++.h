

#ifndef XRAYLIB_PLUSPLUS_H
#define XRAYLIB_PLUSPLUS_H

#include <xraylib.h>
#include <stdexcept>
#include <new> 

namespace xrlpp {
    void process_error(xrl_error *error) {
        if (!error)
            return;
        switch (error->code) {
            case XRL_ERROR_MEMORY:
                throw std::bad_alloc();
            case XRL_ERROR_INVALID_ARGUMENT:
                throw std::invalid_argument(error->message);
            default:
                throw std::runtime_error(error->message);
        }
    }

    double AtomicWeight(int Z) {
        xrl_error *error = nullptr;
        double rv = ::AtomicWeight(Z, &error);
        process_error(error);
        return rv;
    }
}

#endif
