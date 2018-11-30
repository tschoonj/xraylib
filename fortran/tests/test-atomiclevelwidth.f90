PROGRAM test_atomiclevelwidth

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
REAL (C_DOUBLE) :: width

width = AtomicLevelWidth(26, K_SHELL, error)
CALL assert(ABS(width - 1.19E-3_C_DOUBLE) < 1E-6_C_DOUBLE)
CALL assert(.NOT. ASSOCIATED(error))

width = AtomicLevelWidth(92, N7_SHELL, error)
CALL assert(ABS(width - 0.31E-3_C_DOUBLE) < 1E-8_C_DOUBLE);
CALL assert(.NOT. ASSOCIATED(error))

width = AtomicLevelWidth(185, K_SHELL, error)
CALL assert(width == 0.0_C_DOUBLE)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

width = AtomicLevelWidth(26, -5, error)
CALL assert(width == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

width = AtomicLevelWidth(26, N3_SHELL, error)
CALL assert(width == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

width = AtomicLevelWidth(26, K_SHELL)
CALL assert(ABS(width - 1.19E-3_C_DOUBLE) < 1E-6_C_DOUBLE)

width = AtomicLevelWidth(-1, K_SHELL)
CALL assert(width == 0.0_C_DOUBLE)

ENDPROGRAM test_atomiclevelwidth
