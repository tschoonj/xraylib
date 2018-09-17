PROGRAM test_atomicweight

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
REAL (C_DOUBLE) :: weight

weight = AtomicWeight(26, error)
CALL assert(ABS(weight - 55.850_C_DOUBLE) .LT. 1E-6)
CALL assert(.NOT. ASSOCIATED(error))

weight = AtomicWeight(92, error)
CALL assert(ABS(weight - 238.070_C_DOUBLE) .LT. 1E-6)
CALL assert(.NOT. ASSOCIATED(error))

weight = AtomicWeight(185, error)
CALL assert(weight == 0.0_C_DOUBLE)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

weight = AtomicWeight(48)
CALL assert(weight == 112.41_C_DOUBLE)

weight = AtomicWeight(-1)
CALL assert(weight == 0.0_C_DOUBLE)


ENDPROGRAM test_atomicweight
