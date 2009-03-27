;
; This file is the initialization of the IDL interface to xraylib 
;
; Usage: ; From your IDL session, enter:
;    @~/.xraylib/idl/xraylib.pro 
;
; Then use the xraylib functions, for example:
; print,'Testing xraylib: K-edge energy [keV] for Z=26 is: ',edgeenergy(26,0)
;
;
AVOGNUM = 0.602252        ; Avogadro number (mol-1 * barn-1 * cm2) 
KEV2ANGST = 12.398520     ; keV to angstrom-1 conversion factor 
MEC2 = 511.0034           ; electron rest mass (keV) 
RE2 = 0.07940775          ; square of classical electron radius (barn)

XRAYLIB_DIR=getenv('XRAYLIB_DIR')
!path=!path+':'+XRAYLIB_DIR+'/idl'

.run xraylib_lines
.run xraylib_shells
.compile xraylib_help

KA_LINE = 0
KB_LINE = 1
LA_LINE = 2
LB_LINE = 3
      
F1_TRANS   = 0    
F12_TRANS  = 1     
F13_TRANS  = 2    
FP13_TRANS = 3     
F23_TRANS  = 4    



LINKIMAGE,'XRayInit',XRAYLIB_DIR+'/idl/libxrl.so',0,'XRayInit',$
           MAX_ARGS=0

LINKIMAGE,'SetHardExit',XRAYLIB_DIR+'/idl/libxrl.so',0,$
          'IDL_SetHardExit', MAX_ARGS=1, MIN_ARGS=1

LINKIMAGE,'SetExitStatus',XRAYLIB_DIR+'/idl/libxrl.so',0,$
          'IDL_SetExitStatus', MAX_ARGS=1, MIN_ARGS=1

LINKIMAGE,'GetExitStatus',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_GetExitStatus', MAX_ARGS=0

LINKIMAGE,'AtomicWeight',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_AtomicWeight', MAX_ARGS=1, MIN_ARGS=1

LINKIMAGE,'CS_Total',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_Total', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CS_Photo',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_Photo', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CS_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_Rayl', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CS_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_Compt', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CS_FluorLine',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_FluorLine', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'CS_KN',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_KN', MAX_ARGS=1, MIN_ARGS=1

LINKIMAGE,'CSb_Total',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CSb_Total', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CSb_Photo',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CSb_Photo', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CSb_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CSb_Rayl', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CSb_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CSb_Compt', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CSb_FluorLine',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CSb_FluorLine', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCS_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCS_Rayl', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCS_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCS_Compt', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCS_Thoms',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCS_Thoms', MAX_ARGS=1, MIN_ARGS=1

LINKIMAGE,'DCS_KN',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCS_KN', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'DCSb_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSb_Rayl', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCSb_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSb_Compt', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCSP_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSP_Rayl', MAX_ARGS=4, MIN_ARGS=4

LINKIMAGE,'DCSP_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSP_Compt', MAX_ARGS=4, MIN_ARGS=4

LINKIMAGE,'DCSP_Thoms',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSP_Thoms', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'DCSP_KN',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSP_KN', MAX_ARGS=3, MIN_ARGS=3

LINKIMAGE,'DCSPb_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSPb_Rayl', MAX_ARGS=4, MIN_ARGS=4

LINKIMAGE,'DCSPb_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_DCSPb_Compt', MAX_ARGS=4, MIN_ARGS=4

LINKIMAGE,'FF_Rayl',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_FF_Rayl', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'SF_Compt',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_SF_Compt', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'MomentTransf',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_MomentTransf', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'ComptonEnergy',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_ComptonEnergy', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'EdgeEnergy',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_EdgeEnergy', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'LineEnergy',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_LineEnergy', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'FluorYield',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_FluorYield', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'JumpFactor',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_JumpFactor', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'RadRate',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_RadRate', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CosKronTransProb',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CosKronTransProb', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'Fi',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_Fi', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'Fii',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_Fii', MAX_ARGS=2, MIN_ARGS=2

LINKIMAGE,'CS_Photo_Total',XRAYLIB_DIR+'/idl/libxrl.so',1,$
          'IDL_CS_Photo_Total', MAX_ARGS=2, MIN_ARGS=2
XRayInit

