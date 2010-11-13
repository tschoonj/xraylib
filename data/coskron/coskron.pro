

;parses MShellConstants.dat


line = ''
data_template = {FM12: 0.0D, FM13: 0.0D, FM14: 0.0D, FM15: 0.0D, FM23: 0.0D, FM24: 0.0D, FM25: 0.0D, FM34: 0.0D, FM35: 0.0D, FM45:0.0D}

my_data = REPLICATE(data_template,109)

shell = ''



OPENR,lun,'MShellConstants.dat',/GET_LUN

WHILE (NOT EOF(lun)) DO BEGIN
	READF,lun,line
	SWITCH STRMID(line,0,1) OF
		'#': BEGIN
			CASE STRMID(line,0,4) OF
				'#S 1': shell = 'M1'
				'#S 2': shell = 'M2'
				'#S 3': shell = 'M3'
				'#S 4': shell = 'M4'
				'#S 5': shell = 'M5'
				ELSE:
			ENDCASE
			BREAK
		END
		'1':
		'2':
		'3':
		'4':
		'5':
		'6':
		'7':
		'8':
		'9': BEGIN
			parsed_line = DOUBLE(STRSPLIT(line,/EXTRACT))	
			;PRINT,'Element: ',FIX(parsed_line[0])
			CASE shell OF
				'M1': BEGIN
					my_data[FIX(parsed_line[0])-1].FM12 = parsed_line[1]
					my_data[FIX(parsed_line[0])-1].FM13 = parsed_line[2]
					my_data[FIX(parsed_line[0])-1].FM14 = parsed_line[3]
					my_data[FIX(parsed_line[0])-1].FM15 = parsed_line[4]
				END
				'M2': BEGIN
					my_data[FIX(parsed_line[0])-1].FM23 = parsed_line[1]
					my_data[FIX(parsed_line[0])-1].FM24 = parsed_line[2]
					my_data[FIX(parsed_line[0])-1].FM25 = parsed_line[3]
				END
				'M3': BEGIN
					my_data[FIX(parsed_line[0])-1].FM34 = parsed_line[1]
					my_data[FIX(parsed_line[0])-1].FM35 = parsed_line[2]
				END
				'M4': BEGIN
					my_data[FIX(parsed_line[0])-1].FM45 = parsed_line[3]
				END
				'M5': BEGIN
					GOTO,M5SWITCH
				END
			ENDCASE
		END
		ELSE:
	ENDSWITCH
ENDWHILE

M5SWITCH:

FREE_LUN,lun


;copy coskronKL.dat to coskron.dat
FILE_COPY, 'coskronKL.dat','coskron.dat',/OVERWRITE

;add M-shell information
OPENW,lun,'coskron.dat',/GET_LUN,/APPEND
FOR i=0,N_ELEMENTS(my_data)-1 DO BEGIN
	IF (my_data[i].FM12 GT 0.0D) THEN PRINTF,lun,i+1,'FM12',my_data[i].FM12,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM13 GT 0.0D) THEN PRINTF,lun,i+1,'FM13',my_data[i].FM13,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM14 GT 0.0D) THEN PRINTF,lun,i+1,'FM14',my_data[i].FM14,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM15 GT 0.0D) THEN PRINTF,lun,i+1,'FM15',my_data[i].FM15,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM23 GT 0.0D) THEN PRINTF,lun,i+1,'FM23',my_data[i].FM23,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM24 GT 0.0D) THEN PRINTF,lun,i+1,'FM24',my_data[i].FM24,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM25 GT 0.0D) THEN PRINTF,lun,i+1,'FM25',my_data[i].FM25,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM34 GT 0.0D) THEN PRINTF,lun,i+1,'FM34',my_data[i].FM34,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM35 GT 0.0D) THEN PRINTF,lun,i+1,'FM35',my_data[i].FM35,FORMAT='(I3,5X,A,5X,E13.6)'
	IF (my_data[i].FM45 GT 0.0D) THEN PRINTF,lun,i+1,'FM45',my_data[i].FM45,FORMAT='(I3,5X,A,5X,E13.6)'
ENDFOR



FREE_LUN,lun








END
