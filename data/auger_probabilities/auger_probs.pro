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

STOP



OPENW, lun, '../auger_rates.dat', /GET_LUN

FOR i=0,N_ELEMENTS(data)-1 DO BEGIN
	FOR j=0,N_ELEMENTS((*data[i]).names)-1 DO BEGIN
		dims=SIZE((*data[i]).data,/DIMENSIONS)
		FOR k=0,dims[1]-1 DO BEGIN
			PRINTF,lun,k+1,(*data[i]).names[j],(*data[i]).data[j,k],FORMAT='(I3,4X,A,E16.8)'
		ENDFOR
	ENDFOR
ENDFOR
FREE_LUN,lun


FREE_LUN, lun




END
