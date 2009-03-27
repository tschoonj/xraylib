#include "xrayglob.h"
#include "xraylib.h"

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                            Atomic Weight                         //
//                                                                  //
//          Z : atomic number                                       //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float AtomicWeight(int Z)
{
  float atomic_weight;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function AtomicWeight");
    return 0;
  }

  atomic_weight = AtomicWeight_arr[Z];
  if (atomic_weight < 0.) {
    ErrorExit("Atomic Weight not available in function AtomicWeight");
    return 0;
  }
  return atomic_weight;
}

