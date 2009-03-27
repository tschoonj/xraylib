


F2003 = g95
CC = gcc


SRC_F2003 = xrlexample3.F90
SRC_C = xrlexample1.c

OBJ_F2003 = xrlexample3
OBJ_C = xrlexample1

SRC_PERL = xrlexample2.pl

#CFLAGS = -m32
#FFLAGS = -m32

C:	$(SRC_C)
	rm -f $(OBJ_C)
	$(CC) $(CFLAGS) -I${XRAYLIB_DIR}/include -L${XRAYLIB_DIR}/lib -o $(OBJ_C) $(SRC_C) -lxrl -lm
	./$(OBJ_C)


F2003:	$(SRC_F2003)
	rm -f $(OBJ_F2003)
	$(F2003) $(FFLAGS) -I${XRAYLIB_DIR}/include -L${XRAYLIB_DIR}/lib -o $(OBJ_F2003) $(SRC_F2003) -lxrl
	./$(OBJ_F2003)
perl: $(SRC_PERL)
	./$(SRC_PERL)

all:
	make clean
	make C
	make F2003
	make perl
clean:
	rm -f $(OBJ_C ) $(OBJ_F2003)
