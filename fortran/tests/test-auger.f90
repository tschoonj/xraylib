PROGRAM test_auger

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
REAL (C_DOUBLE) :: rate, yield

rate = AugerRate(82, K_L3M5_AUGER, error)
CALL assert(ABS(rate - 0.004573193387_C_DOUBLE) < 1E-6_C_DOUBLE)
CALL assert(.NOT. ASSOCIATED(error))

rate = AugerRate(82, L3_M4N7_AUGER, error)
CALL assert(ABS(rate - 0.0024327572005_C_DOUBLE) < 1E-6_C_DOUBLE)
CALL assert(.NOT. ASSOCIATED(error))

rate = AugerRate(-35, L3_M4N7_AUGER, error)
CALL assert(rate == 0.0_C_DOUBLE)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

rate = AugerRate(82, M4_M5Q3_AUGER + 1, error)
CALL assert(rate == 0.0_C_DOUBLE)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

rate = AugerRate(62, L3_M4N7_AUGER, error)
CALL assert(rate == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

yield = AugerYield(82, K_SHELL, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(yield - (1.0_C_DOUBLE - FluorYield(82, K_SHELL))) < 1E-6_C_DOUBLE)

yield = AugerYield(82, M3_SHELL, error)
CALL assert(ABS(yield - 0.1719525_C_DOUBLE) < 1E-6_C_DOUBLE)
CALL assert(.NOT. ASSOCIATED(error))

yield = AugerYield(-35, K_SHELL, error)
CALL assert(yield == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

yield = AugerYield(82, N2_SHELL, error)
CALL assert(yield == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

yield = AugerYield(26, M5_SHELL, error)
CALL assert(yield == 0.0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

ENDPROGRAM test_auger
