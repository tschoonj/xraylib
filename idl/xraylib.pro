
;Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



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

XRAYLIB_DIR=getenv('XRAYLIB_DIR')
!path=!path+':'+XRAYLIB_DIR+'/idl'
;!dlm_path=!dlm_path+':'+XRAYLIB_DIR+'/idl'

COMMON XRAYLIB,AVOGNUM,$
	KEV2ANGST,$
	MEC2,$
	RE2,$
	KA_LINE,$
	KB_LINE,$
	LA_LINE,$
	LB_LINE,$
	F1_TRANS,$
	F12_TRANS,$
	F13_TRANS,$
	FP13_TRANS,$
	F23_TRANS,$
	KL1_LINE,$
	KL2_LINE,$
	KL3_LINE,$
	KM1_LINE,$
	KM2_LINE,$
	KM3_LINE,$
	KM4_LINE,$
	KM5_LINE,$
	KN1_LINE,$
	KN2_LINE,$
	KN3_LINE,$
	KN4_LINE,$
	KN5_LINE,$
	L1M1_LINE,$
	L1M2_LINE,$
	L1M3_LINE,$
	L1M4_LINE,$
	L1M5_LINE,$
	L1N1_LINE,$
	L1N2_LINE,$
	L1N3_LINE,$
	L1N4_LINE,$
	L1N5_LINE,$
	L1N6_LINE,$
	L1N7_LINE,$
	L2M1_LINE,$
	L2M2_LINE,$
	L2M3_LINE,$
	L2M4_LINE,$
	L2M5_LINE,$
	L2N1_LINE,$
	L2N2_LINE,$
	L2N3_LINE,$
	L2N4_LINE,$
	L2N5_LINE,$
	L2N6_LINE,$
	L2N7_LINE,$
	L3M1_LINE,$
	L3M2_LINE,$
	L3M3_LINE,$
	L3M4_LINE,$
	L3M5_LINE,$
	L3N1_LINE,$
	L3N2_LINE,$
	L3N3_LINE,$
	L3N4_LINE,$
	L3N5_LINE,$
	L3N6_LINE,$
	L3N7_LINE,$
	K_SHELL,$
	L1_SHELL,$
	L2_SHELL,$
	L3_SHELL,$
	M1_SHELL,$
	M2_SHELL,$
	M3_SHELL,$
	M4_SHELL,$
	M5_SHELL,$
	N1_SHELL,$
	N2_SHELL,$
	N3_SHELL,$
	N4_SHELL,$
	N5_SHELL,$
	N6_SHELL,$
	N7_SHELL,$
	O1_SHELL,$
	O2_SHELL,$
	O3_SHELL,$
	O4_SHELL,$
	O5_SHELL,$
	O6_SHELL,$
	O7_SHELL,$
	P1_SHELL,$
	P2_SHELL,$
	P3_SHELL,$
	P4_SHELL,$
	P5_SHELL


AVOGNUM = 0.602252        ; Avogadro number (mol-1 * barn-1 * cm2) 
KEV2ANGST = 12.398520     ; keV to angstrom-1 conversion factor 
MEC2 = 511.0034           ; electron rest mass (keV) 
RE2 = 0.07940775          ; square of classical electron radius (barn)

KA_LINE = 0
KB_LINE = 1
LA_LINE = 2
LB_LINE = 3
      
F1_TRANS   = 0    
F12_TRANS  = 1     
F13_TRANS  = 2    
FP13_TRANS = 3     
F23_TRANS  = 4    



.run xraylib_lines
.run xraylib_shells
.compile xraylib_help



;XRayInit
