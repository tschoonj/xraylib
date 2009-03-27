#include "xrayglob.h"
#include "xraylib.h"

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Jump Ratio                                    //
//                                                                  //
//          Z : atomic number                                       //
//          shell :                                                 //
//            K_SHELL  0                                            //
//            L1_SHELL 1                                            //
//            L2_SHELL 2                                            //
//            L3_SHELL 3                                            //
//            M1_SHELL 4                                            //
//            M2_SHELL 5                                            //
//            M3_SHELL 6                                            //
//            M4_SHELL 7                                            //
//            M5_SHELL 8                                            //
//             .......                                              //
//                                                                  //
//////////////////////////////////////////////////////////////////////
      
float JumpFactor(int Z, int shell)
{
  float jump_factor;

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function JumpFactor");
    return 0;
  }

  if (shell<0 || shell>=SHELLNUM) {
    ErrorExit("Shell not available in function JumpFactor");
    return 0;
  }

  jump_factor = JumpFactor_arr[Z][shell];
  if (jump_factor < 0.) {
    ErrorExit("Shell not available in function JumpFactor");
    return 0;
  }

  return jump_factor;
}

                          
                          
