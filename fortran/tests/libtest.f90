MODULE libtest

INTERFACE
        SUBROUTINE test_exit(exit_status) BIND(C, NAME='exit')
                USE, INTRINSIC :: ISO_C_BINDING
                IMPLICIT NONE
                INTEGER (C_INT), INTENT(IN), VALUE :: exit_status
        ENDSUBROUTINE test_exit
ENDINTERFACE

CONTAINS

SUBROUTINE ASSERT(condition)
        USE, INTRINSIC :: ISO_FORTRAN_ENV
        IMPLICIT NONE
        LOGICAL, INTENT(IN) :: condition
        INTEGER, SAVE :: counter = 1
        WRITE(output_unit, '(A, I5)') 'assert call ', counter
        counter = counter + 1
        IF (.NOT. condition) CALL test_exit(1)
ENDSUBROUTINE ASSERT
ENDMODULE libtest
