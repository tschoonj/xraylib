;Copyright (c) 2009, Tom Schoonjans
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

;Procedure designed to convert the Kissel PE cross section data into a form that is more easy to read by xraylib


FUNCTION read_total,lun

line=''
total_pe={energies:PTR_NEW() , cs: PTR_NEW() , cs2: PTR_NEW() }
nlines=0L


;read junk
FOR i=0,6 DO READF,lun,line

REPEAT BEGIN
 READF,lun,line
 ;abort when necessary
 IF (STRCMP(line,' *** END OF DATA ***',STRLEN(' *** END OF DATA ***'))) THEN BREAK
 
 values=STRSPLIT(STRTRIM(line,1),/EXTRACT)
 IF (nlines EQ 0) THEN BEGIN
	temp_energies=[ALOG(DOUBLE(values[0])) ]
 	temp_cs=[ALOG(DOUBLE(values[1]))]
 ENDIF ELSE BEGIN
	temp_energies=[temp_energies,ALOG(DOUBLE(values[0]))]	
	temp_cs=[temp_cs,ALOG(DOUBLE(values[1]))]
 ENDELSE
 nlines += 1
ENDREP UNTIL 0

;calculate the second derivatives
temp_cs2=DERIV(temp_energies,DERIV(temp_energies,temp_cs))
temp_cs2[WHERE(temp_cs2 lt -1.0 OR temp_cs2 gt 1.0)] = 0.0
 
total_pe.energies=PTR_NEW(temp_energies)
total_pe.cs=PTR_NEW(temp_cs)
total_pe.cs2=PTR_NEW(temp_cs2)
 
RETURN,total_pe

END

FUNCTION read_partial_ss,lun,type

line=''
nlines=0L

search=STRCOMPRESS('*BLOCK:'+type,/REMOVE)




REPEAT BEGIN
 READF,lun,line
ENDREP UNTIL (STRCMP(search,line,STRLEN(search)))

FOR i=0,14 DO READF,lun,line


REPEAT BEGIN
 READF,lun,line
 ;abort when necessary
 IF (STRCMP(line,' *** END OF DATA ***',STRLEN(' *** END OF DATA ***'))) THEN BREAK
 
 values=STRSPLIT(STRTRIM(line,1),/EXTRACT)
 IF (nlines EQ 0) THEN BEGIN
	temp_energies=[ALOG(DOUBLE(values[0])) ]
 	temp_cs=[ALOG(DOUBLE(values[1]))]
 ENDIF ELSE BEGIN
	temp_energies=[temp_energies,ALOG(DOUBLE(values[0]))]	
	temp_cs=[temp_cs,ALOG(DOUBLE(values[1]))]
 ENDELSE
 nlines += 1
ENDREP UNTIL 0


;calculate the second derivatives
temp_cs2=DERIV(temp_energies,DERIV(temp_energies,temp_cs))
temp_cs2[WHERE(temp_cs2 lt -1.0 OR temp_cs2 gt 1.0)] = 0.0

ptrs=PTRARR(3)


ptrs[0]=PTR_NEW(temp_energies)
ptrs[1]=PTR_NEW(temp_cs)
ptrs[2]=PTR_NEW(temp_cs2)

RETURN,ptrs

END



FUNCTION read_partial,lun,config

partial_pe={   $ 
	K : PTRARR(3),$ 
	L1: PTRARR(3), L2: PTRARR(3), L3: PTRARR(3),$
	M1: PTRARR(3), M2: PTRARR(3), M3: PTRARR(3), M4: PTRARR(3), M5: PTRARR(3),$
	N1: PTRARR(3), N2: PTRARR(3), N3: PTRARR(3), N4: PTRARR(3), N5: PTRARR(3), N6: PTRARR(3), N7: PTRARR(3),$
	O1: PTRARR(3), O2: PTRARR(3), O3: PTRARR(3), O4: PTRARR(3), O5: PTRARR(3), O6: PTRARR(3), O7: PTRARR(3),$
	P1: PTRARR(3), P2: PTRARR(3) ,P3: PTRARR(3) ,P4: PTRARR(3), P5: PTRARR(3),$
	Q1: PTRARR(3), Q2: PTRARR(3), Q3: PTRARR(3)}

	IF (config.K NE 0.0) THEN partial_pe.K=read_partial_ss(lun,'K')
	IF (config.L1 NE 0.0) THEN partial_pe.L1=read_partial_ss(lun,'L1') $
		ELSE partial_pe.L1[0]=PTR_NEW(-1)
	IF (config.L2 NE 0.0) THEN partial_pe.L2=read_partial_ss(lun,'L2') $
		ELSE partial_pe.L2[0]=PTR_NEW(-1)
	IF (config.L3 NE 0.0) THEN partial_pe.L3=read_partial_ss(lun,'L3') $
		ELSE partial_pe.L3[0]=PTR_NEW(-1)
	IF (config.M1 NE 0.0) THEN partial_pe.M1=read_partial_ss(lun,'M1') $
		ELSE partial_pe.M1[0]=PTR_NEW(-1)
	IF (config.M2 NE 0.0) THEN partial_pe.M2=read_partial_ss(lun,'M2') $
		ELSE partial_pe.M2[0]=PTR_NEW(-1)
	IF (config.M3 NE 0.0) THEN partial_pe.M3=read_partial_ss(lun,'M3') $
		ELSE partial_pe.M3[0]=PTR_NEW(-1)
	IF (config.M4 NE 0.0) THEN partial_pe.M4=read_partial_ss(lun,'M4') $
		ELSE partial_pe.M4[0]=PTR_NEW(-1)
	IF (config.M5 NE 0.0) THEN partial_pe.M5=read_partial_ss(lun,'M5') $
		ELSE partial_pe.M5[0]=PTR_NEW(-1)
	IF (config.N1 NE 0.0) THEN partial_pe.N1=read_partial_ss(lun,'N1') $
		ELSE partial_pe.N1[0]=PTR_NEW(-1)
	IF (config.N2 NE 0.0) THEN partial_pe.N2=read_partial_ss(lun,'N2') $
		ELSE partial_pe.N2[0]=PTR_NEW(-1)
	IF (config.N3 NE 0.0) THEN partial_pe.N3=read_partial_ss(lun,'N3') $
		ELSE partial_pe.N3[0]=PTR_NEW(-1)
	IF (config.N4 NE 0.0) THEN partial_pe.N4=read_partial_ss(lun,'N4') $
		ELSE partial_pe.N4[0]=PTR_NEW(-1)
	IF (config.N5 NE 0.0) THEN partial_pe.N5=read_partial_ss(lun,'N5') $
		ELSE partial_pe.N5[0]=PTR_NEW(-1)
	IF (config.N6 NE 0.0) THEN partial_pe.N6=read_partial_ss(lun,'N6') $
		ELSE partial_pe.N6[0]=PTR_NEW(-1)
	IF (config.N7 NE 0.0) THEN partial_pe.N7=read_partial_ss(lun,'N7') $
		ELSE partial_pe.N7[0]=PTR_NEW(-1)
	IF (config.O1 NE 0.0) THEN partial_pe.O1=read_partial_ss(lun,'O1') $
		ELSE partial_pe.O1[0]=PTR_NEW(-1)
	IF (config.O2 NE 0.0) THEN partial_pe.O2=read_partial_ss(lun,'O2') $
		ELSE partial_pe.O2[0]=PTR_NEW(-1)
	IF (config.O3 NE 0.0) THEN partial_pe.O3=read_partial_ss(lun,'O3') $
		ELSE partial_pe.O3[0]=PTR_NEW(-1)
	IF (config.O4 NE 0.0) THEN partial_pe.O4=read_partial_ss(lun,'O4') $
		ELSE partial_pe.O4[0]=PTR_NEW(-1)
	IF (config.O5 NE 0.0) THEN partial_pe.O5=read_partial_ss(lun,'O5') $
		ELSE partial_pe.O5[0]=PTR_NEW(-1)
	IF (config.O6 NE 0.0) THEN partial_pe.O6=read_partial_ss(lun,'O6') $
		ELSE partial_pe.O6[0]=PTR_NEW(-1)
	IF (config.O7 NE 0.0) THEN partial_pe.O7=read_partial_ss(lun,'O7') $
		ELSE partial_pe.O7[0]=PTR_NEW(-1)
	IF (config.P1 NE 0.0) THEN partial_pe.P1=read_partial_ss(lun,'P1') $
		ELSE partial_pe.P1[0]=PTR_NEW(-1)
	IF (config.P2 NE 0.0) THEN partial_pe.P2=read_partial_ss(lun,'P2') $
		ELSE partial_pe.P2[0]=PTR_NEW(-1)
	IF (config.P3 NE 0.0) THEN partial_pe.P3=read_partial_ss(lun,'P3') $
		ELSE partial_pe.P3[0]=PTR_NEW(-1)
	IF (config.P4 NE 0.0) THEN partial_pe.P4=read_partial_ss(lun,'P4') $
		ELSE partial_pe.P4[0]=PTR_NEW(-1)
	IF (config.P5 NE 0.0) THEN partial_pe.P5=read_partial_ss(lun,'P5') $
		ELSE partial_pe.P5[0]=PTR_NEW(-1)
	IF (config.Q1 NE 0.0) THEN partial_pe.Q1=read_partial_ss(lun,'Q1') $
		ELSE partial_pe.Q1[0]=PTR_NEW(-1)
	IF (config.Q2 NE 0.0) THEN partial_pe.Q2=read_partial_ss(lun,'Q2') $
		ELSE partial_pe.Q2[0]=PTR_NEW(-1)
	IF (config.Q3 NE 0.0) THEN partial_pe.Q3=read_partial_ss(lun,'Q3') $
		ELSE partial_pe.Q3[0]=PTR_NEW(-1)




RETURN,partial_pe

END


FUNCTION ss_config,lun,binding_energies
;determines the subshell configuration

ss={    K : 0.0,$ 
	L1: 0.0, L2: 0.0, L3: 0.0,$
	M1: 0.0, M2: 0.0, M3: 0.0, M4: 0.0, M5: 0.0,$
	N1: 0.0, N2: 0.0, N3: 0.0, N4: 0.0, N5: 0.0, N6: 0.0, N7: 0.0,$
	O1: 0.0, O2: 0.0, O3: 0.0, O4: 0.0, O5: 0.0, O6: 0.0, O7: 0.0,$
	P1: 0.0, P2: 0.0 ,P3: 0.0 ,P4: 0.0, P5: 0.0,$
	Q1: 0.0, Q2: 0.0, Q3: 0.0}
binding_energies=REPLICATE(ss,1)

line=''


REPEAT BEGIN
 READF,lun,line
ENDREP UNTIL (STRCMP('*BLOCK:CONFIGURATION',line,STRLEN('*BLOCK:CONFIGURATION')))

;read 12 lines 

FOR i=0,11 DO READF,lun,line

REPEAT BEGIN
READF,lun,line

IF (STRLEN(STRTRIM(line,2)) EQ 0) THEN CONTINUE

IF (STRCMP(line,' *** END OF DATA ***',STRLEN(' *** END OF DATA ***'))) THEN BREAK


values=STRSPLIT(STRTRIM(line,1),/EXTRACT)

CASE 1 OF 
	;K shell
	(values[0] EQ 1): BEGIN 
		ss.K=FLOAT(values[4])
		binding_energies.K=FLOAT(values[5])
		END
	;L shells
	(values[0] EQ 2) AND (values[1] EQ -1): BEGIN
		ss.L1=FLOAT(values[4])
		binding_energies.L1=FLOAT(values[5])
		END
	(values[0] EQ 2) AND (values[1] EQ  1): BEGIN
		ss.L2=FLOAT(values[4])
		binding_energies.L2=FLOAT(values[5])
		END
	(values[0] EQ 2) AND (values[1] EQ -2): BEGIN
		ss.L3=FLOAT(values[4])
		binding_energies.L3=FLOAT(values[5])
		END
	;M shells
	(values[0] EQ 3) AND (values[1] EQ -1): BEGIN
		ss.M1=FLOAT(values[4])
		binding_energies.M1=FLOAT(values[5])
		END
	(values[0] EQ 3) AND (values[1] EQ  1): BEGIN
		ss.M2=FLOAT(values[4])
		binding_energies.M2=FLOAT(values[5])
		END
	(values[0] EQ 3) AND (values[1] EQ -2): BEGIN
		ss.M3=FLOAT(values[4])
		binding_energies.M3=FLOAT(values[5])
		END
	(values[0] EQ 3) AND (values[1] EQ  2): BEGIN
		ss.M4=FLOAT(values[4])
		binding_energies.M4=FLOAT(values[5])
		END
	(values[0] EQ 3) AND (values[1] EQ -3): BEGIN
		ss.M5=FLOAT(values[4])
		binding_energies.M5=FLOAT(values[5])
		END
	;N shells
	(values[0] EQ 4) AND (values[1] EQ -1): BEGIN
		ss.N1=FLOAT(values[4])
		binding_energies.N1=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ  1): BEGIN
		ss.N2=FLOAT(values[4])
		binding_energies.N2=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ -2): BEGIN
		ss.N3=FLOAT(values[4])
		binding_energies.N3=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ  2): BEGIN
		ss.N4=FLOAT(values[4])
		binding_energies.N4=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ -3): BEGIN
		ss.N5=FLOAT(values[4])
		binding_energies.N5=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ  3): BEGIN
		ss.N6=FLOAT(values[4])
		binding_energies.N6=FLOAT(values[5])
		END
	(values[0] EQ 4) AND (values[1] EQ -4): BEGIN
		ss.N7=FLOAT(values[4])
		binding_energies.N7=FLOAT(values[5])
		END
	;O shells		
	(values[0] EQ 5) AND (values[1] EQ -1): BEGIN
		ss.O1=FLOAT(values[4])
		binding_energies.O1=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ  1): BEGIN
		ss.O2=FLOAT(values[4])
		binding_energies.O2=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ -2): BEGIN
		ss.O3=FLOAT(values[4])
		binding_energies.O3=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ  2): BEGIN
		ss.O4=FLOAT(values[4])
		binding_energies.O4=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ -3): BEGIN
		ss.O5=FLOAT(values[4])
		binding_energies.O5=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ  3): BEGIN
		ss.O6=FLOAT(values[4])
		binding_energies.O6=FLOAT(values[5])
		END
	(values[0] EQ 5) AND (values[1] EQ -4): BEGIN
		ss.O7=FLOAT(values[4])
		binding_energies.O7=FLOAT(values[5])
		END
	;P shells
	(values[0] EQ 6) AND (values[1] EQ -1): BEGIN
		ss.P1=FLOAT(values[4])
		binding_energies.P1=FLOAT(values[5])
		END
	(values[0] EQ 6) AND (values[1] EQ  1): BEGIN
		ss.P2=FLOAT(values[4])
		binding_energies.P2=FLOAT(values[5])
		END
	(values[0] EQ 6) AND (values[1] EQ -2): BEGIN
		ss.P3=FLOAT(values[4])
		binding_energies.P3=FLOAT(values[5])
		END
	(values[0] EQ 6) AND (values[1] EQ  2): BEGIN
		ss.P4=FLOAT(values[4])
		binding_energies.P4=FLOAT(values[5])
		END
	(values[0] EQ 6) AND (values[1] EQ -3): BEGIN
		ss.P5=FLOAT(values[4])
		binding_energies.P5=FLOAT(values[5])
		END
	;Q shells
	(values[0] EQ 7) AND (values[1] EQ -1): BEGIN
		ss.Q1=FLOAT(values[4])
		binding_energies.Q1=FLOAT(values[5])
		END
	(values[0] EQ 7) AND (values[1] EQ  1): BEGIN
		ss.Q2=FLOAT(values[4])
		binding_energies.Q2=FLOAT(values[5])
		END
	(values[0] EQ 7) AND (values[1] EQ -2): BEGIN
		ss.Q3=FLOAT(values[4])
		binding_energies.Q3=FLOAT(values[5])
		END
	

ENDCASE



ENDREP UNTIL 0




RETURN,ss

END


PRO kissel



;testfile='092_pe0sl'

;get_lun,lun

;openr,lun,testfile

;first read the total cross sections
;total_pe=read_total(lun)

; then read the electronic configuration

;config=ss_config(lun)

;and last but not least, read the partial cross sections

;partial_pe=read_partial(lun,config)


;plot,*total_pe.energies,*total_pe.cs,/ylog,XRANGE=[0.0,10.0]
;oplot,*(partial_pe.K[0]),*(partial_pe.K[1])*config.K
;print,'Fe.M1: ',config.M1
;print,'Fe.M5: ',config.M5


;close,lun
;free_lun,lun

;Determine all the files

files=FILE_SEARCH('0*')
print,files

GET_LUN,lunw
OPENW,lunw,'../kissel_pe.dat'


FOR i=0,N_ELEMENTS(files)-1 DO BEGIN

	GET_LUN,lunr
	OPENR,lunr,files[i]
	;read the total cross sections
	total_pe=read_total(lunr)
	;read the electronic configuration
	config=ss_config(lunr,binding_energies)
	;read the partial cross sections
	partial_pe=read_partial(lunr,config)
	
	CLOSE,lunr
	FREE_LUN,lunr

	;write everything to the output file
	;first the total cross sections
	PRINTF,lunw,N_ELEMENTS(*total_pe.energies)
	FOR j=0,N_ELEMENTS(*total_pe.energies)-1 DO BEGIN
		PRINTF,lunw,(*total_pe.energies)[j],(*total_pe.cs)[j],(*total_pe.cs2)[j]
	ENDFOR

	;print electronic configuration	
	PRINTF,lunw,config.K
	PRINTF,lunw,config.L1
	PRINTF,lunw,config.L2
	PRINTF,lunw,config.L3
	PRINTF,lunw,config.M1
	PRINTF,lunw,config.M2
	PRINTF,lunw,config.M3
	PRINTF,lunw,config.M4
	PRINTF,lunw,config.M5
	PRINTF,lunw,config.N1
	PRINTF,lunw,config.N2
	PRINTF,lunw,config.N3
	PRINTF,lunw,config.N4
	PRINTF,lunw,config.N5
	PRINTF,lunw,config.N6
	PRINTF,lunw,config.N7
	PRINTF,lunw,config.O1
	PRINTF,lunw,config.O2
	PRINTF,lunw,config.O3
	PRINTF,lunw,config.O4
	PRINTF,lunw,config.O5
	PRINTF,lunw,config.O6
	PRINTF,lunw,config.O7
	PRINTF,lunw,config.P1
	PRINTF,lunw,config.P2
	PRINTF,lunw,config.P3
	PRINTF,lunw,config.P4
	PRINTF,lunw,config.P5
	PRINTF,lunw,config.Q1
	PRINTF,lunw,config.Q2
	PRINTF,lunw,config.Q3
	
	;print partial cross sections
	IF (config.K NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.K[0])
		PRINTF,lunw,binding_energies.K
		FOR j=0,N_ELEMENTS(*partial_pe.K[0])-1 DO $
			PRINTF,lunw,(*partial_pe.K[0])[j],$
			(*partial_pe.K[1])[j],$
			(*partial_pe.K[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.L1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.L1[0])
		PRINTF,lunw,binding_energies.L1
		FOR j=0,N_ELEMENTS(*partial_pe.L1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.L1[0])[j],$
			(*partial_pe.L1[1])[j],$
			(*partial_pe.L1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.L2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.L2[0])
		PRINTF,lunw,binding_energies.L2
		FOR j=0,N_ELEMENTS(*partial_pe.L2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.L2[0])[j],$
			(*partial_pe.L2[1])[j],$
			(*partial_pe.L2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.L3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.L3[0])
		PRINTF,lunw,binding_energies.L3
		FOR j=0,N_ELEMENTS(*partial_pe.L3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.L3[0])[j],$
			(*partial_pe.L3[1])[j],$
			(*partial_pe.L3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.M1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.M1[0])
		PRINTF,lunw,binding_energies.M1
		FOR j=0,N_ELEMENTS(*partial_pe.M1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.M1[0])[j],$
			(*partial_pe.M1[1])[j],$
			(*partial_pe.M1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.M2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.M2[0])
		PRINTF,lunw,binding_energies.M2
		FOR j=0,N_ELEMENTS(*partial_pe.M2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.M2[0])[j],$
			(*partial_pe.M2[1])[j],$
			(*partial_pe.M2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.M3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.M3[0])
		PRINTF,lunw,binding_energies.M3
		FOR j=0,N_ELEMENTS(*partial_pe.M3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.M3[0])[j],$
			(*partial_pe.M3[1])[j],$
			(*partial_pe.M3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.M4 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.M4[0])
		PRINTF,lunw,binding_energies.M4
		FOR j=0,N_ELEMENTS(*partial_pe.M4[0])-1 DO $
			PRINTF,lunw,(*partial_pe.M4[0])[j],$
			(*partial_pe.M4[1])[j],$
			(*partial_pe.M4[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.M5 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.M5[0])
		PRINTF,lunw,binding_energies.M5
		FOR j=0,N_ELEMENTS(*partial_pe.M5[0])-1 DO $
			PRINTF,lunw,(*partial_pe.M5[0])[j],$
			(*partial_pe.M5[1])[j],$
			(*partial_pe.M5[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N1[0])
		PRINTF,lunw,binding_energies.N1
		FOR j=0,N_ELEMENTS(*partial_pe.N1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N1[0])[j],$
			(*partial_pe.N1[1])[j],$
			(*partial_pe.N1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N2[0])
		PRINTF,lunw,binding_energies.N2
		FOR j=0,N_ELEMENTS(*partial_pe.N2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N2[0])[j],$
			(*partial_pe.N2[1])[j],$
			(*partial_pe.N2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N3[0])
		PRINTF,lunw,binding_energies.N3
		FOR j=0,N_ELEMENTS(*partial_pe.N3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N3[0])[j],$
			(*partial_pe.N3[1])[j],$
			(*partial_pe.N3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N4 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N4[0])
		PRINTF,lunw,binding_energies.N4
		FOR j=0,N_ELEMENTS(*partial_pe.N4[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N4[0])[j],$
			(*partial_pe.N4[1])[j],$
			(*partial_pe.N4[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N5 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N5[0])
		PRINTF,lunw,binding_energies.N5
		FOR j=0,N_ELEMENTS(*partial_pe.N5[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N5[0])[j],$
			(*partial_pe.N5[1])[j],$
			(*partial_pe.N5[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N6 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N6[0])
		PRINTF,lunw,binding_energies.N6
		FOR j=0,N_ELEMENTS(*partial_pe.N6[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N6[0])[j],$
			(*partial_pe.N6[1])[j],$
			(*partial_pe.N6[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.N7 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.N7[0])
		PRINTF,lunw,binding_energies.N7
		FOR j=0,N_ELEMENTS(*partial_pe.N7[0])-1 DO $
			PRINTF,lunw,(*partial_pe.N7[0])[j],$
			(*partial_pe.N7[1])[j],$
			(*partial_pe.N7[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O1[0])
		PRINTF,lunw,binding_energies.O1
		FOR j=0,N_ELEMENTS(*partial_pe.O1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O1[0])[j],$
			(*partial_pe.O1[1])[j],$
			(*partial_pe.O1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O2[0])
		PRINTF,lunw,binding_energies.O2
		FOR j=0,N_ELEMENTS(*partial_pe.O2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O2[0])[j],$
			(*partial_pe.O2[1])[j],$
			(*partial_pe.O2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O3[0])
		PRINTF,lunw,binding_energies.O3
		FOR j=0,N_ELEMENTS(*partial_pe.O3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O3[0])[j],$
			(*partial_pe.O3[1])[j],$
			(*partial_pe.O3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O4 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O4[0])
		PRINTF,lunw,binding_energies.O4
		FOR j=0,N_ELEMENTS(*partial_pe.O4[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O4[0])[j],$
			(*partial_pe.O4[1])[j],$
			(*partial_pe.O4[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O5 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O5[0])
		PRINTF,lunw,binding_energies.O5
		FOR j=0,N_ELEMENTS(*partial_pe.O5[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O5[0])[j],$
			(*partial_pe.O5[1])[j],$
			(*partial_pe.O5[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O6 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O6[0])
		PRINTF,lunw,binding_energies.O6
		FOR j=0,N_ELEMENTS(*partial_pe.O6[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O6[0])[j],$
			(*partial_pe.O6[1])[j],$
			(*partial_pe.O6[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.O7 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.O7[0])
		PRINTF,lunw,binding_energies.O7
		FOR j=0,N_ELEMENTS(*partial_pe.O7[0])-1 DO $
			PRINTF,lunw,(*partial_pe.O7[0])[j],$
			(*partial_pe.O7[1])[j],$
			(*partial_pe.O7[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.P1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.P1[0])
		PRINTF,lunw,binding_energies.P1
		FOR j=0,N_ELEMENTS(*partial_pe.P1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.P1[0])[j],$
			(*partial_pe.P1[1])[j],$
			(*partial_pe.P1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.P2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.P2[0])
		PRINTF,lunw,binding_energies.P2
		FOR j=0,N_ELEMENTS(*partial_pe.P2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.P2[0])[j],$
			(*partial_pe.P2[1])[j],$
			(*partial_pe.P2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.P3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.P3[0])
		PRINTF,lunw,binding_energies.P3
		FOR j=0,N_ELEMENTS(*partial_pe.P3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.P3[0])[j],$
			(*partial_pe.P3[1])[j],$
			(*partial_pe.P3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.P4 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.P4[0])
		PRINTF,lunw,binding_energies.P4
		FOR j=0,N_ELEMENTS(*partial_pe.P4[0])-1 DO $
			PRINTF,lunw,(*partial_pe.P4[0])[j],$
			(*partial_pe.P4[1])[j],$
			(*partial_pe.P4[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.P5 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.P5[0])
		PRINTF,lunw,binding_energies.P5
		FOR j=0,N_ELEMENTS(*partial_pe.P5[0])-1 DO $
			PRINTF,lunw,(*partial_pe.P5[0])[j],$
			(*partial_pe.P5[1])[j],$
			(*partial_pe.P5[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.Q1 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.Q1[0])
		PRINTF,lunw,binding_energies.Q1
		FOR j=0,N_ELEMENTS(*partial_pe.Q1[0])-1 DO $
			PRINTF,lunw,(*partial_pe.Q1[0])[j],$
			(*partial_pe.Q1[1])[j],$
			(*partial_pe.Q1[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.Q2 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.Q2[0])
		PRINTF,lunw,binding_energies.Q2
		FOR j=0,N_ELEMENTS(*partial_pe.Q2[0])-1 DO $
			PRINTF,lunw,(*partial_pe.Q2[0])[j],$
			(*partial_pe.Q2[1])[j],$
			(*partial_pe.Q2[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE
	IF (config.Q3 NE 0.0) THEN BEGIN
	 	PRINTF,lunw,N_ELEMENTS(*partial_pe.Q3[0])
		PRINTF,lunw,binding_energies.Q3
		FOR j=0,N_ELEMENTS(*partial_pe.Q3[0])-1 DO $
			PRINTF,lunw,(*partial_pe.Q3[0])[j],$
			(*partial_pe.Q3[1])[j],$
			(*partial_pe.Q3[2])[j]
	ENDIF ELSE BEGIN
		PRINTF,lunw,0
	ENDELSE


ENDFOR




CLOSE,lunw
FREE_LUN,lunw


PRINT,'kissel_pe.dat was succesfully created'



END
