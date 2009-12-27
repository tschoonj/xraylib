
;Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



;
; This file takes care of the initialization of the IDL interface to xraylib 
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
	KN45_LINE,$
	KO23_LINE,$
	KO45_LINE,$
	KP23_LINE,$
	L1L3_LINE,$
	L1M45_LINE,$
	L1N45_LINE,$
	L1O23_LINE,$
	L1P23_LINE,$
	L2O1_LINE,$
	L2O4_LINE,$
	L2P1_LINE,$
	L3O1_LINE,$
	L3O45_LINE,$
	L3P1_LINE,$
	M1M2_LINE,$
	M1M3_LINE,$
	M1M4_LINE,$
	M1M5_LINE,$
	M1N1_LINE,$
	M1N2_LINE,$
	M1N3_LINE,$
	M1N4_LINE,$
	M1N5_LINE,$
	M1N6_LINE,$
	M1N7_LINE,$
	M1O1_LINE,$
	M1O2_LINE,$
	M1O3_LINE,$
	M1O4_LINE,$
	M1O5_LINE,$
	M1P2_LINE,$
	M1P3_LINE,$
	M2M3_LINE,$
	M2M4_LINE,$
	M2M5_LINE,$
	M2N1_LINE,$
	M2N2_LINE,$
	M2N3_LINE,$
	M2N4_LINE,$
	M2N5_LINE,$
	M2N6_LINE,$
	M2N7_LINE,$
	M2O1_LINE,$
	M2O2_LINE,$
	M2O3_LINE,$
	M2O4_LINE,$
	M2O5_LINE,$
	M2O6_LINE,$
	M2P1_LINE,$
	M2P4_LINE,$
	M3M4_LINE,$
	M3M5_LINE,$
	M3N1_LINE,$
	M3N2_LINE,$
	M3N3_LINE,$
	M3N4_LINE,$
	M3N5_LINE,$
	M3N6_LINE,$
	M3N7_LINE,$
	M3O1_LINE,$
	M3O2_LINE,$
	M3O3_LINE,$
	M3O4_LINE,$
	M3O5_LINE,$
	M3O6_LINE,$
	M3P2_LINE,$
	M3Q1_LINE,$
	M4M5_LINE,$
	M4N1_LINE,$
	M4N2_LINE,$
	M4N3_LINE,$
	M4N4_LINE,$
	M4N5_LINE,$
	M4N6_LINE,$
	M4N7_LINE,$
	M4O1_LINE,$
	M4O2_LINE,$
	M4O3_LINE,$
	M4O4_LINE,$
	M4O5_LINE,$
	M4O6_LINE,$
	M4P1_LINE,$
	M4P2_LINE,$
	M4P3_LINE,$
	M5N1_LINE,$
	M5N2_LINE,$
	M5N3_LINE,$
	M5N4_LINE,$
	M5N5_LINE,$
	M5N6_LINE,$
	M5N7_LINE,$
	M5O1_LINE,$
	M5O2_LINE,$
	M5O3_LINE,$
	M5O4_LINE,$
	M5O5_LINE,$
	M5O6_LINE,$
	M5P1_LINE,$
	M5P3_LINE,$
	M5P4_LINE,$
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
