FUNCTION read_auger_file_section,lun



line=''
;#N
READF, lun, line
print,line
;#L
READF, lun, line
print,line
names=STRSPLIT(line,/EXTRACT)
print,names
names = names[2:N_ELEMENTS(names)-1]
;check out name of the third transition
IF (N_ELEMENTS(names) GT 1) THEN BEGIN
IF (STRMID(names[1],0,1) EQ 'K') THEN BEGIN
	names[0] = 'K-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'L1') THEN BEGIN
	names[0] = 'L1-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'L2') THEN BEGIN
	names[0] = 'L2-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'L3') THEN BEGIN
	names[0] = 'L3-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'M1') THEN BEGIN
	names[0] = 'M1-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'M2') THEN BEGIN
	names[0] = 'M2-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'M3') THEN BEGIN
	names[0] = 'M3-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'M4') THEN BEGIN
	names[0] = 'M4-TOTAL'
ENDIF ELSE IF(STRMID(names[1],0,2) EQ 'M5') THEN BEGIN
	names[0] = 'M5-TOTAL'
ENDIF
ENDIF ELSE BEGIN
	names[0] = 'M5-TOTAL'
ENDELSE

data = []

WHILE (NOT EOF(lun)) DO BEGIN
	READF, lun, line
	line=STRTRIM(line,1)
	IF (line EQ '') THEN BREAK
	IF (STRMID(line,0,1) EQ '#') THEN CONTINUE
	data_prov = DOUBLE(STRSPLIT(line,/EXTRACT))
	;IF (data_prov[1] GT 0.0 AND N_ELEMENTS(data_prov) GT 2) THEN BEGIN
	;	data_prov[2:N_ELEMENTS(data_prov)-1] /= data_prov[1]
	;ENDIF
	data = [data, data_prov[1:N_ELEMENTS(data_prov)-1] ]
ENDWHILE

data = REFORM(data, N_ELEMENTS(names),N_ELEMENTS(data)/N_ELEMENTS(names),/OVERWRITE)

;rv = hash('names',names,'data',data)

rv = {names:names, data:data}


return,PTR_NEW(rv)

END


FUNCTION read_auger_file,filename

OPENR, lun, filename, /GET_LUN

line = ''
sections = []
WHILE (NOT EOF(lun)) DO BEGIN
	READF, lun, line
	line=STRTRIM(line,1)
	IF (STRMID(line, 0, 2) EQ '#S') THEN sections = [sections, read_auger_file_section(lun)]
ENDWHILE


FREE_LUN, lun

RETURN,sections

END




PRO auger_probs


files = ['EADL97_KShellNonradiativeRates.dat','EADL97_LShellNonradiativeRates.dat','EADL97_MShellNonradiativeRates.dat']


data = []


FOREACH file, files DO BEGIN
	print,file
	data = [data, read_auger_file(file)]
ENDFOREACH




OPENW, lun, '../auger_rates.dat', /GET_LUN
OPENW, lun2, '../../include/xraylib-auger.h', /get_lun
PRINTF, lun2, '/* Copyright (c) 2009-2013 Tom Schoonjans'
PRINTF, lun2, 'All rights reserved.'
PRINTF, lun2, ''
PRINTF, lun2, 'Redistribution and use in source and binary forms, with or without'
PRINTF, lun2, 'modification, are permitted provided that the following conditions are met:'
PRINTF, lun2, '    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.'
PRINTF, lun2, '    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.'
PRINTF, lun2, '    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.'
PRINTF, lun2, ''
PRINTF, lun2, 'THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'
PRINTF, lun2, '*/'

PRINTF, lun2, ''
PRINTF, lun2, '#ifndef XRAYLIB_AUGER_H'
PRINTF, lun2, '#define XRAYLIB_AUGER_H'
PRINTF, lun2, ''

macro = 0


forbidden = ['O8', 'O9', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']


FOR i=0,N_ELEMENTS(data)-1 DO BEGIN
	dims=SIZE((*data[i]).data,/DIMENSIONS)
	FOR j=0,N_ELEMENTS((*data[i]).names)-1 DO BEGIN
		name = (*data[i]).names[j]
		matched = 0
		FOREACH forbid, forbidden DO BEGIN
			IF (STRPOS(name, forbid) NE -1) THEN BEGIN
				matched = 1
				BREAK
			ENDIF
		ENDFOREACH
		IF (matched) THEN BEGIN
			;calculate sum
			sum = TOTAL((*data[i]).data[j,*])
			IF (sum GT 0.0) THEN PRINT, 'name ',name, ' has positive sum: ',sum
			CONTINUE
		ENDIF

		;convert name to acceptable macro
		IF (STRPOS(name,'TOTAL') EQ -1) THEN BEGIN
			pos = STRPOS(name, '-')
			STRPUT,name,'_',pos
			name += '_AUGER'

			PRINTF, lun2, '#define ',name,macro++,FORMAT='(A,1X,A,X,I4)'
		ENDIF
		FOR k=0,dims[1]-1 DO BEGIN
			PRINTF,lun,k+1,(*data[i]).names[j],(*data[i]).data[j,k],FORMAT='(I3,4X,A,E16.8)'
		ENDFOR
	ENDFOR
ENDFOR
PRINTF, lun2, ''
PRINTF, lun2, '#endif'
FREE_LUN,lun
FREE_LUN,lun2





END
