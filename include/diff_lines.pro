FUNCTION read_lines_h,file

OPENR,lun,file,/GET_LUN

ntrans=0L

line=''

WHILE (NOT EOF(lun)) DO BEGIN
	READF,lun,line	
	IF (STRMID(line,0,1) EQ '#' ) THEN BEGIN
		splitted =strsplit(line,/EXTRACT)
		IF (ntrans EQ 0L) THEN rv = [splitted[1]] ELSE rv= [rv,splitted[1]]
		ntrans++
	ENDIF

ENDWHILE

FREE_LUN,lun

RETURN,rv

END

PRO diff_lines,file1,file2

transitions1 = read_lines_h(file1)
transitions2 = read_lines_h(file2)


	FOR i=0L,N_ELEMENTS(transitions1)-1 DO BEGIN
		IF (WHERE(transitions2 EQ transitions1[i]) EQ -1) THEN PRINT,transitions1[i]+' is contained in '+file1+' but is missing from '+file2
	ENDFOR

	FOR i=0L,N_ELEMENTS(transitions2)-1 DO BEGIN
		IF (WHERE(transitions1 EQ transitions2[i]) EQ -1) THEN PRINT,transitions2[i]+' is contained in '+file2+' but is missing from '+file1
	ENDFOR




END
