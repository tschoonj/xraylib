;program designed to generate lines.h based on the lines that are present in both radrade.dat and fluor_lines.dat



PRO read_fluor_or_rad_dat,filename,all_transitions

length=FILE_LINES(filename)

all_transitions=strarr(length)

OPENR,lun,filename,/GET_LUN

line=''

FOR i=0L,length-1 DO BEGIN
	READF,lun,line
	splitted=strsplit(line,/EXTRACT)
	all_transitions[i]=splitted[1]
ENDFOR

FREE_LUN,lun

END


PRO write_lines_h,lines_h


read_fluor_or_rad_dat,'../data/fluor_lines.dat',fluor_dat_transitions
read_fluor_or_rad_dat,'../data/radrate.dat',radrates_dat_transitions


all_transitions = [fluor_dat_transitions,radrates_dat_transitions]

all_transitions_sort_uniq = all_transitions[UNIQ(all_transitions,SORT(all_transitions))]

OPENW,lun,lines_h,/GET_LUN

FOR i=0L,N_ELEMENTS(all_transitions_sort_uniq)-1 DO BEGIN
	PRINTF,lun,STRCOMPRESS('#define '+all_transitions_sort_uniq[i]+'_LINE '+STRING(-1*(i+1),FORMAT='(I4)'))
ENDFOR

FREE_LUN,lun


END

