PROGRAM test_compoundparser

USE, INTRINSIC :: ISO_C_BINDING
USE, INTRINSIC :: ISO_FORTRAN_ENV
USE :: xraylib
USE :: libtest
IMPLICIT NONE

TYPE (xrl_error), POINTER :: error => NULL()
CHARACTER (KIND=C_CHAR,LEN=3) :: symbol
INTEGER :: Z
TYPE (compoundData), POINTER :: cd

CALL assert(ASSOCIATED(CompoundParser("C19H29COOH")))
CALL assert(ASSOCIATED(CompoundParser("C12H10")))
CALL assert(ASSOCIATED(CompoundParser("C12H6O2")))
CALL assert(ASSOCIATED(CompoundParser("C6H5Br")))
CALL assert(ASSOCIATED(CompoundParser("C3H4OH(COOH)3")))
CALL assert(ASSOCIATED(CompoundParser("HOCH2CH2OH")))
CALL assert(ASSOCIATED(CompoundParser("C5H11NO2")))
CALL assert(ASSOCIATED(CompoundParser("CH3CH(CH3)CH3")))
CALL assert(ASSOCIATED(CompoundParser("NH2CH(C4H5N2)COOH")))
CALL assert(ASSOCIATED(CompoundParser("H2O")))
CALL assert(ASSOCIATED(CompoundParser("Ca5(PO4)3F")))
CALL assert(ASSOCIATED(CompoundParser("Ca5(PO4)3OH")))
CALL assert(ASSOCIATED(CompoundParser("Ca5.522(PO4.48)3OH")))
CALL assert(ASSOCIATED(CompoundParser("Ca5.522(PO.448)3OH")))

cd => CompoundParser('H2SO4')
CALL assert(cd%nElements == 3)
CALL assert(ABS(cd%molarMass - 98.09_C_DOUBLE) < 1E-6_C_DOUBLE)
CALL assert(ALL(cd%Elements - [1, 8, 16] == 0))
CALL assert(ALL(ABS(cd%massFractions - [0.02059333265368539_C_DOUBLE, &
        0.6524620246712203_C_DOUBLE, 0.32694464267509427_C_DOUBLE]) &
        < 1E-6_C_DOUBLE))
CALL assert(ALL(ABS(cd%nAtoms - [2.0_C_DOUBLE, 4.0_C_DOUBLE, &
        1.0_C_DOUBLE]) < 1E-6_C_DOUBLE))
CALL assert(ABS(cd%nAtomsAll - 7.0) < 1E-6)
DEALLOCATE(cd)

CALL assert(.NOT. ASSOCIATED(CompoundParser("CuI2ww")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("0C")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("2O")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("13Li")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("2(NO3)")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("H(2)")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Ba(12)")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Cr(5)3")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Pb(13)2")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au(22)11")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au11(H3PO4)2)")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au11(H3PO4))2")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au(11(H3PO4))2")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Ca5.522(PO.44.8)3OH")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Ba[12]")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Auu1")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("AuL1")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("  ")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("\t")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("\n")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au L1")))
CALL assert(.NOT. ASSOCIATED(CompoundParser("Au\tFe")))

CALL assert(SymbolToAtomicNumber("Fe", error) == 26)
CALL assert(.NOT. ASSOCIATED(error))

CALL assert(SymbolToAtomicNumber("Uu", error) == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT)
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

symbol = AtomicNumberToSymbol(26, error)
CALL assert(symbol == "Fe")
CALL assert(.NOT. ASSOCIATED(error))

symbol = AtomicNumberToSymbol(-2, error)
CALL assert(LEN_TRIM(symbol) == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT);
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

symbol = AtomicNumberToSymbol(108, error)
CALL assert(LEN_TRIM(symbol) == 0)
CALL assert(ASSOCIATED(error))
CALL assert(error%code == XRL_ERROR_INVALID_ARGUMENT);
WRITE (output_unit, '(A,A)') 'Error message: ', TRIM(error%message)
DEALLOCATE(error)

DO Z = 1, 107
        symbol = AtomicNumberToSymbol(Z)
        CALL assert(LEN_TRIM(symbol) > 0)
        CALL assert(SymbolToAtomicNumber(symbol) == Z)
ENDDO
ENDPROGRAM test_compoundparser
