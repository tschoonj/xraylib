PROGRAM test_refractive_indices

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE(xrl_error), POINTER :: error => NULL()
REAL (C_DOUBLE) :: re, im
COMPLEX (C_DOUBLE) :: cplx

re = Refractive_Index_Re("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(re - 0.999763450676632_C_DOUBLE) < 1E-9_C_DOUBLE)

im = Refractive_Index_Im("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(im - 4.021660592312145e-05_C_DOUBLE) < 1E-9_C_DOUBLE)

cplx = Refractive_Index("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(REAL(cplx, C_DOUBLE) - 0.999763450676632_C_DOUBLE) < 1E-9_C_DOUBLE)
CALL assert(ABS(AIMAG(cplx) - 4.021660592312145e-05_C_DOUBLE) < 1E-9_C_DOUBLE)

re = Refractive_Index_Re("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(ABS(re - 0.999763450676632_C_DOUBLE) < 1E-9_C_DOUBLE)

im = Refractive_Index_Im("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(ABS(im - 4.021660592312145e-05_C_DOUBLE) < 1E-9_C_DOUBLE)

cplx = Refractive_Index("H2O", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(ABS(REAL(cplx, C_DOUBLE) - 0.999763450676632_C_DOUBLE) < 1E-9_C_DOUBLE)
CALL assert(ABS(AIMAG(cplx) - 4.021660592312145e-05_C_DOUBLE) < 1E-9_C_DOUBLE)

re = Refractive_Index_Re("Air, Dry (near sea level)", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(re - 0.999782559048_C_DOUBLE) < 1E-9_C_DOUBLE)

im = Refractive_Index_Im("Air, Dry (near sea level)", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(im - 0.000035578193_C_DOUBLE) < 1E-9_C_DOUBLE)

cplx = Refractive_Index("Air, Dry (near sea level)", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(REAL(cplx, C_DOUBLE) - 0.999782559048_C_DOUBLE) < 1E-9_C_DOUBLE)
CALL assert(ABS(AIMAG(cplx) - 0.000035578193_C_DOUBLE) < 1E-9_C_DOUBLE)

re = Refractive_Index_Re("Air, Dry (near sea level)", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(re - 0.999999737984_C_DOUBLE) < 1E-12_C_DOUBLE)

im = Refractive_Index_Im("Air, Dry (near sea level)", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(im - 0.000000042872_C_DOUBLE) < 1E-12_C_DOUBLE)

cplx = Refractive_Index("Air, Dry (near sea level)", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(REAL(cplx, C_DOUBLE) - 0.999999737984_C_DOUBLE) < 1E-12_C_DOUBLE)
CALL assert(ABS(AIMAG(cplx) - 0.000000042872_C_DOUBLE) < 1E-12_C_DOUBLE)

cplx = Refractive_Index("Air, Dry (near sea level)", 1.0_C_DOUBLE, -1.0_C_DOUBLE, error)
CALL assert(.NOT. ASSOCIATED(error))
CALL assert(ABS(REAL(cplx, C_DOUBLE) - re) < 1E-12_C_DOUBLE)
CALL assert(ABS(AIMAG(cplx) - im) < 1E-12_C_DOUBLE)

! bad input
re = Refractive_Index_Re("", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(re == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

im = Refractive_Index_Im("", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(im == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

cplx = Refractive_Index("", 1.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

re = Refractive_Index_Re("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(re == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

im = Refractive_Index_Im("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(im == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

cplx = Refractive_Index("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

re = Refractive_Index_Re("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(re == 0.0)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

im = Refractive_Index_Im("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(im == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

cplx = Refractive_Index("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE, error)
CALL assert(ASSOCIATED(error))
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

re = Refractive_Index_Re("", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(re == 0.0_C_DOUBLE)

im = Refractive_Index_Im("", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(im == 0.0_C_DOUBLE)

cplx = Refractive_Index("", 1.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)

re = Refractive_Index_Re("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(re == 0.0_C_DOUBLE)

im = Refractive_Index_Im("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(im == 0.0_C_DOUBLE)

cplx = Refractive_Index("H2O", 0.0_C_DOUBLE, 1.0_C_DOUBLE)
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)

re = Refractive_Index_Re("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE)
CALL assert(re == 0.0)

im = Refractive_Index_Im("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE)
CALL assert(im == 0.0_C_DOUBLE)

cplx = Refractive_Index("H2O", 1.0_C_DOUBLE, 0.0_C_DOUBLE)
CALL assert(REAL(cplx, C_DOUBLE) == 0.0_C_DOUBLE .AND. AIMAG(cplx) == 0.0_C_DOUBLE)

ENDPROGRAM test_refractive_indices

