#include "xrayglob.h"
#include "xraylib.h"

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Coster-Kronig transition probability          //
//          Z : atomic number                                       //
//          transition type :                                       //
//            F1_TRANS  0                                           //
//            F12_TRANS 1                                           //
//            F13_TRANS 2                                           //
//            FP13_TRANS 3                                          //
//            F23_TRANS 4                                           //
//                                                                  //
//////////////////////////////////////////////////////////////////////
      
float CosKronTransProb(int Z, int trans)
{
  float trans_prob;

  if (Z<1 || Z>ZMAX){
    ErrorExit("Z out of range in function CosKronTransProb");
    return 0;
  }

  if (trans<0 || trans>=TRANSNUM) {
    ErrorExit("Transition not available in function CosKronTransProb");
    return 0;
  }

  trans_prob = CosKron_arr[Z][trans];
  if (trans_prob < 0.) {
    ErrorExit("Transition not available in function CosKronTransProb");
    return 0;
  }

  return trans_prob;
}

                          
                          
