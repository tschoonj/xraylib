/*
Copyright (c) 2015-2021, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package com.github.tschoonj.xraylib;

import java.io.DataInputStream;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.numbers.complex.Complex;


/** 
 * This is the main class of the xraylib package, containing all static methods. 
 * 
 * If an invalid argument has been passed to any of these methods,
 * an @see IllegalArgumentException will be thrown.
 * 
 * @author Tom Schoonjans (Tom.Schoonjans@diamond.ac.uk)
 * @since 3.2.0
 */
public class Xraylib {

  static {
    try {
      XRayInit();
    }
    catch (Exception e){
      e.printStackTrace();
      System.exit(1);
    }
  }

  /* predefined error messages  */
  public static final String Z_OUT_OF_RANGE = "Z out of range";
  public static final String NEGATIVE_ENERGY = "Energy must be strictly positive";
  public static final String NEGATIVE_DENSITY = "Density must be strictly positive";
  public static final String NEGATIVE_Q = "q must be positive";
  public static final String NEGATIVE_PZ = "pz must be positive";
  public static final String INVALID_SHELL = "Invalid shell for this atomic number";
  public static final String INVALID_LINE = "Invalid line for this atomic number";
  public static final String INVALID_CK = "Invalid Coster-Kronig transition for this atomic number";
  public static final String INVALID_AUGER = "Invalid Auger transition macro for this atomic number";
  public static final String UNKNOWN_SHELL = "Unknown shell macro provided";
  public static final String UNKNOWN_LINE = "Unknown line macro provided";
  public static final String UNKNOWN_CK = "Unknown Coster-Kronig transition macro provided";
  public static final String UNKNOWN_AUGER = "Unknown Auger transition macro provided";
  public static final String UNAVAILABLE_JUMP_FACTOR = "Jump factor unavailable for element and shell";
  public static final String UNAVAILABLE_FLUOR_YIELD = "Fluorescence yield unavailable for atomic number and shell";
  public static final String TOO_LOW_EXCITATION_ENERGY = "The excitation energy too low to excite the shell";
  public static final String UNAVAILABLE_PHOTO_CS = "Photoionization cross section unavailable for atomic number and energy";
  public static final String UNAVAILABLE_RAD_RATE = "Radiative rate unavailable for this atomic number and line macro";
  public static final String UNAVAILABLE_CK = "Coster-Kronig transition probability unavailable for this atomic number and transition macro";
  public static final String UNKNOWN_COMPOUND = "Compound is not a valid chemical formula and is not present in the NIST compound database";
  public static final String MALLOC_ERROR = "Could not allocate memory";
  public static final String INVALID_MILLER = "Miller indices cannot all be zero";
  public static final String NEGATIVE_DEBYE_FACTOR = "Debye-Waller factor must be strictly positive";
  public static final String CRYSTAL_NULL = "Crystal cannot be NULL";
  public static final String SPLINT_X_TOO_LOW = "Spline extrapolation is not allowed";
  public static final String SPLINT_X_TOO_HIGH = "Spline extrapolation is not allowed";
  public static final String LININTERP_X_TOO_LOW = "Linear extrapolation is not allowed";
  public static final String LININTERP_X_TOO_HIGH = "Linear extrapolation is not allowed";

  protected static String readString(ByteBuffer byte_buffer) {
    ArrayList<Byte> al = new ArrayList<>();
    while (true) {
      byte my_char = byte_buffer.get();
      if (my_char == 0)
        break;
      al.add(my_char);
    }
    byte[] temp = new byte[al.size()];
    Iterator<Byte> iterator = al.iterator();
    for (int i = 0 ; i < temp.length ; i++) {
      temp[i] = iterator.next().byteValue();
    }
    return new String(temp, StandardCharsets.US_ASCII);
  }

  protected static double[][][] readDoubleArrayOfArraysOfArrays(int[][] N, ByteBuffer byte_buffer) throws BufferUnderflowException {
    double[][][] rv = new double[N.length][][];

    for (int i = 0 ; i < N.length ; i++) {
      rv[i] = readDoubleArrayOfArrays(N[i], byte_buffer);
    }

    return rv;
  }

  protected static double[][] readDoubleArrayOfArrays(int[] N, ByteBuffer byte_buffer) throws BufferUnderflowException {
    double [][] rv = new double[N.length][];

    for (int i = 0 ; i < N.length ; i++) {
      if (N[i] <= 0) {
        rv[i] = null;
        continue;
      }
      rv[i] = readDoubleArray(N[i], byte_buffer);
    }

    return rv;
  }

  protected static double[] readDoubleArray(int n, ByteBuffer byte_buffer) throws BufferUnderflowException {
    //System.out.println("readDoubleArray n: " + n);

    double[] rv = new double[n];

    for (int i = 0 ; i < n ; i++) {
      try {
        rv[i] = byte_buffer.getDouble();
        //System.out.println("rv["+i+"]: "+ rv[i]);
      }
      catch (BufferUnderflowException e) {
        throw new BufferUnderflowException();
      }
    }

    return rv;
  }

  protected static int[] readIntArray(int n, ByteBuffer byte_buffer) throws BufferUnderflowException {
    int[] rv = new int[n];

    for (int i = 0 ; i < n ; i++) {
      try {
        rv[i] = byte_buffer.getInt();
        //System.out.println("rv["+i+"]: "+ rv[i]);
      }
      catch (BufferUnderflowException e) {
        throw new BufferUnderflowException();
      }
    }

    return rv;
  }

  protected static int[][] arrayReshape(int[] array, int m, int n) {
    if (m*n != array.length)
      throw new RuntimeException("new array dimensions incompatible with old ones");

    int[][] rv = new int[m][n];

    int index_old = 0;
    for (int i = 0 ; i < m ; i++) {
      for (int j = 0 ; j < n ; j++) {
        rv[i][j] = array[index_old++];
      }
    }
    return rv;
  }

  private static void XRayInit() throws Exception {
    try (
      DataInputStream inputStream = new DataInputStream(Xraylib.class.getClassLoader().getResourceAsStream("xraylib.dat"));
      ) {
      
      int bytes_total = inputStream.available();
      byte[] bytes = new byte[bytes_total];
      inputStream.readFully(bytes);
      ByteBuffer byte_buffer = ByteBuffer.wrap(bytes);
      byte_buffer.order(ByteOrder.LITTLE_ENDIAN);
      ZMAX = byte_buffer.getInt();
      SHELLNUM = byte_buffer.getInt();
      SHELLNUM_K = byte_buffer.getInt();
      SHELLNUM_A = byte_buffer.getInt();
      TRANSNUM = byte_buffer.getInt();
      LINENUM = byte_buffer.getInt();
      AUGERNUM = byte_buffer.getInt();
      RE2 = byte_buffer.getDouble();
      MEC2 = byte_buffer.getDouble();
      AVOGNUM = byte_buffer.getDouble();
      KEV2ANGST = byte_buffer.getDouble();
      R_E = byte_buffer.getDouble();
      //System.out.println("ZMAX: " + ZMAX);
      //System.out.println("SHELLNUM: " + SHELLNUM);
      //System.out.println("TRANSNUM: " + TRANSNUM);

      AtomicWeight_arr = readDoubleArray(ZMAX + 1, byte_buffer);
      ElementDensity_arr = readDoubleArray(ZMAX + 1, byte_buffer);
      EdgeEnergy_arr = readDoubleArray((ZMAX + 1) * SHELLNUM, byte_buffer);
      AtomicLevelWidth_arr = readDoubleArray((ZMAX + 1) * SHELLNUM, byte_buffer);
      LineEnergy_arr = readDoubleArray((ZMAX + 1) * LINENUM, byte_buffer);
      FluorYield_arr = readDoubleArray((ZMAX + 1) * SHELLNUM, byte_buffer);
      JumpFactor_arr = readDoubleArray((ZMAX + 1) * SHELLNUM, byte_buffer);
      CosKron_arr = readDoubleArray((ZMAX + 1) * TRANSNUM, byte_buffer);
      RadRate_arr = readDoubleArray((ZMAX + 1) * LINENUM, byte_buffer);

      NE_Photo_arr = readIntArray(ZMAX + 1, byte_buffer);
      E_Photo_arr = readDoubleArrayOfArrays(NE_Photo_arr, byte_buffer);
      CS_Photo_arr = readDoubleArrayOfArrays(NE_Photo_arr, byte_buffer);
      CS_Photo_arr2 = readDoubleArrayOfArrays(NE_Photo_arr, byte_buffer);

      NE_Rayl_arr = readIntArray(ZMAX + 1, byte_buffer);
      E_Rayl_arr = readDoubleArrayOfArrays(NE_Rayl_arr, byte_buffer);
      CS_Rayl_arr = readDoubleArrayOfArrays(NE_Rayl_arr, byte_buffer);
      CS_Rayl_arr2 = readDoubleArrayOfArrays(NE_Rayl_arr, byte_buffer);

      NE_Compt_arr = readIntArray(ZMAX + 1, byte_buffer);
      E_Compt_arr = readDoubleArrayOfArrays(NE_Compt_arr, byte_buffer);
      CS_Compt_arr = readDoubleArrayOfArrays(NE_Compt_arr, byte_buffer);
      CS_Compt_arr2 = readDoubleArrayOfArrays(NE_Compt_arr, byte_buffer);

      NE_Energy_arr = readIntArray(ZMAX + 1, byte_buffer);
      E_Energy_arr = readDoubleArrayOfArrays(NE_Energy_arr, byte_buffer);
      CS_Energy_arr = readDoubleArrayOfArrays(NE_Energy_arr, byte_buffer);
      CS_Energy_arr2 = readDoubleArrayOfArrays(NE_Energy_arr, byte_buffer);

      Nq_Rayl_arr = readIntArray(ZMAX +1, byte_buffer);
      q_Rayl_arr = readDoubleArrayOfArrays(Nq_Rayl_arr, byte_buffer);
      FF_Rayl_arr = readDoubleArrayOfArrays(Nq_Rayl_arr, byte_buffer);
      FF_Rayl_arr2 = readDoubleArrayOfArrays(Nq_Rayl_arr, byte_buffer);

      Nq_Compt_arr = readIntArray(ZMAX +1, byte_buffer);
      q_Compt_arr = readDoubleArrayOfArrays(Nq_Compt_arr, byte_buffer);
      SF_Compt_arr = readDoubleArrayOfArrays(Nq_Compt_arr, byte_buffer);
      SF_Compt_arr2 = readDoubleArrayOfArrays(Nq_Compt_arr, byte_buffer);

      NE_Fi_arr = readIntArray(ZMAX +1, byte_buffer);
      E_Fi_arr = readDoubleArrayOfArrays(NE_Fi_arr, byte_buffer);
      Fi_arr = readDoubleArrayOfArrays(NE_Fi_arr, byte_buffer);
      Fi_arr2 = readDoubleArrayOfArrays(NE_Fi_arr, byte_buffer);

      NE_Fii_arr = readIntArray(ZMAX +1, byte_buffer);
      E_Fii_arr = readDoubleArrayOfArrays(NE_Fii_arr, byte_buffer);
      Fii_arr = readDoubleArrayOfArrays(NE_Fii_arr, byte_buffer);
      Fii_arr2 = readDoubleArrayOfArrays(NE_Fii_arr, byte_buffer);

      Electron_Config_Kissel_arr = readDoubleArray((ZMAX + 1) * SHELLNUM_K, byte_buffer);
      EdgeEnergy_Kissel_arr = readDoubleArray((ZMAX + 1) * SHELLNUM_K, byte_buffer);

      NE_Photo_Total_Kissel_arr = readIntArray(ZMAX +1, byte_buffer);

      //NE_Photo_Partial_Kissel_arr = readIntArray((ZMAX + 1) * SHELLNUM_K, byte_buffer);
      int[] temp_arr = readIntArray((ZMAX + 1) * SHELLNUM_K, byte_buffer);
      NE_Photo_Partial_Kissel_arr = arrayReshape(temp_arr, ZMAX+1, SHELLNUM_K);
      E_Photo_Partial_Kissel_arr = readDoubleArrayOfArraysOfArrays(NE_Photo_Partial_Kissel_arr, byte_buffer);
      Photo_Partial_Kissel_arr = readDoubleArrayOfArraysOfArrays(NE_Photo_Partial_Kissel_arr, byte_buffer);
      Photo_Partial_Kissel_arr2 = readDoubleArrayOfArraysOfArrays(NE_Photo_Partial_Kissel_arr, byte_buffer);

      NShells_ComptonProfiles_arr = readIntArray(ZMAX +1, byte_buffer);
      Npz_ComptonProfiles_arr = readIntArray(ZMAX +1, byte_buffer);
      UOCCUP_ComptonProfiles_arr = readDoubleArrayOfArrays(NShells_ComptonProfiles_arr, byte_buffer);
      pz_ComptonProfiles_arr = readDoubleArrayOfArrays(Npz_ComptonProfiles_arr, byte_buffer);
      Total_ComptonProfiles_arr = readDoubleArrayOfArrays(Npz_ComptonProfiles_arr, byte_buffer);
      Total_ComptonProfiles_arr2 = readDoubleArrayOfArrays(Npz_ComptonProfiles_arr, byte_buffer);

      Partial_ComptonProfiles_arr = new double[ZMAX+1][][];
      Partial_ComptonProfiles_arr2 = new double[ZMAX+1][][];

      for (int i = 0 ; i < ZMAX+1 ; i++) {
        if (NShells_ComptonProfiles_arr[i] <= 0) {
          continue;
        }
        Partial_ComptonProfiles_arr[i] = new double[NShells_ComptonProfiles_arr[i]][];
        for (int j = 0 ; j < NShells_ComptonProfiles_arr[i] ; j++) {
	  if (Npz_ComptonProfiles_arr[i] > 0 && UOCCUP_ComptonProfiles_arr[i][j] > 0) {
            Partial_ComptonProfiles_arr[i][j] = readDoubleArray(Npz_ComptonProfiles_arr[i], byte_buffer);
          }
        }
      }
      for (int i = 0 ; i < ZMAX+1 ; i++) {
        if (NShells_ComptonProfiles_arr[i] <= 0) {
          continue;
        }
        Partial_ComptonProfiles_arr2[i] = new double[NShells_ComptonProfiles_arr[i]][];
        for (int j = 0 ; j < NShells_ComptonProfiles_arr[i] ; j++) {
	  if (Npz_ComptonProfiles_arr[i] > 0 && UOCCUP_ComptonProfiles_arr[i][j] > 0) {
            Partial_ComptonProfiles_arr2[i][j] = readDoubleArray(Npz_ComptonProfiles_arr[i], byte_buffer);
          }
        }
      }

      Auger_Yields_arr = readDoubleArray((ZMAX + 1)*SHELLNUM_A, byte_buffer);
      Auger_Rates_arr = readDoubleArray((ZMAX + 1)*AUGERNUM, byte_buffer);

      int nCompoundDataNISTList = byte_buffer.getInt();
      compoundDataNISTList = new compoundDataNIST[nCompoundDataNISTList];
      for (int i = 0 ; i < nCompoundDataNISTList ; i++) {
        compoundDataNISTList[i] = new compoundDataNIST(byte_buffer);
      }

      int nNuclideDataList = byte_buffer.getInt();
      nuclideDataList = new radioNuclideData[nNuclideDataList];
      for (int i = 0 ; i < nNuclideDataList ; i++) {
        nuclideDataList[i] = new radioNuclideData(byte_buffer);
      }

      //crystals 
      int nCrystals = byte_buffer.getInt();
      crystalDataList = new Crystal_Struct[nCrystals];
      for (int i = 0 ; i < nCrystals ; i++) {
        crystalDataList[i] = new Crystal_Struct(byte_buffer);
      }

      // precalculated XRF CS components...
      xrf_cross_sections_constants_full = readDoubleArray((ZMAX + 1) * (M5_SHELL + 1) * (L3_SHELL + 1), byte_buffer);
      xrf_cross_sections_constants_auger_only = readDoubleArray((ZMAX + 1) * (M5_SHELL + 1) * (L3_SHELL + 1), byte_buffer);

      //this should never happen!
      if (byte_buffer.hasRemaining()) {
        throw new RuntimeException("byte_buffer not empty when closing!");
      }
    }
  }

  /** 
   * Returns the @see <a href="https://en.wikipedia.org/wiki/Standard_atomic_weight">standard atomic weight</a>
   * 
   * @param Z The atomic number
   * @return The standard atomic weight (dimensionless)
   */
  public static double AtomicWeight(int Z) {
    double atomic_weight;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    atomic_weight = AtomicWeight_arr[Z];

    if (atomic_weight <= 0.) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    return atomic_weight;
  }

  /** 
   * For a given atomic number, returns the @see <a href="https://en.wikipedia.org/wiki/Density">element density</a>.
   * 
   * @param Z The atomic number
   * @return The element density, expressed in g/cm<sup>3</sup>
   */
  public static double ElementDensity(int Z) {
    double element_density;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    element_density = ElementDensity_arr[Z];

    if (element_density <= 0.) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    return element_density;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding @see <a href="https://en.wikipedia.org/wiki/Absorption_edge">absorption edge energy</a>.
   * 
   * This is also known as the electron binding energy.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The absorption edge energy, expressed in keV
   */
  public static double EdgeEnergy(int Z, int shell) {
    double edge_energy;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell < 0 || shell >= SHELLNUM) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    edge_energy = EdgeEnergy_arr[shell + (Z * SHELLNUM)];

    if (edge_energy <= 0.) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return edge_energy;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding atomic level width.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The atomic level width, expressed in keV
   */
  public static double AtomicLevelWidth(int Z, int shell) {
    double atomic_level_width;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell<0 || shell>=SHELLNUM) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    atomic_level_width = AtomicLevelWidth_arr[Z * SHELLNUM + shell];

    if (atomic_level_width <= 0.) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return atomic_level_width;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding fluorescence yield.
   * 
   * The returned value will be between 0 and 1.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The fluorescence yield (dimensionless)
   */
  public static double FluorYield(int Z, int shell) {
    double fluor_yield;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell < 0 || shell >= SHELLNUM) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    fluor_yield = FluorYield_arr[Z * SHELLNUM + shell];

    if (fluor_yield <= 0.) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return fluor_yield;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding jump factor.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The jump factor (dimensionless)
   */
  public static double JumpFactor(int Z, int shell) {
    double jump_factor;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell < 0 || shell >= SHELLNUM) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    jump_factor = JumpFactor_arr[Z * SHELLNUM + shell];

    if (jump_factor <= 0.) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return jump_factor;
  }

  /** 
   * For a given atomic number and transition, returns the corresponding @see <a href="https://en.wikipedia.org/wiki/Coster%E2%80%93Kronig_transition">Coster-Kronig transition probability</a>.
   * 
   * The returned value will be between 0 and 1.
   * 
   * @param Z The atomic number
   * @param trans A macro identifying the Coster-Kronig transition, such as #FL12_TRANS.
   * @return The Coster-Kronig transition probability (dimensionless)
   */
  public static double CosKronTransProb(int Z, int trans) {
    double trans_prob;

    if (Z < 1 || Z > ZMAX){
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (trans < 1 || trans >= TRANSNUM) {
      throw new IllegalArgumentException(UNKNOWN_CK);
    }

    trans_prob = CosKron_arr[Z * TRANSNUM + trans];

    if (trans_prob <= 0.) {
      throw new IllegalArgumentException(INVALID_CK);
    }

    return trans_prob;
  }

  /** 
   * For a given atomic number and line, returns the corresponding radiative rate.
   *
   * The returned value will be between 0 and 1.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @return The radiative rate (dimensionless)
   */
  public static double RadRate(int Z, int line) {
    double rad_rate, rr;
    int i;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (line == KA_LINE) {
      rr = 0.0;
      for (i= 0 ; i <= 2 ; i++) {
        rr += RadRate_arr[Z * LINENUM + i];
      }
      if (rr == 0.0) {
        throw new IllegalArgumentException(INVALID_LINE);
      }
      return rr;
    }
    else if (line == KB_LINE) {
    /*
     * we assume that RR(Ka)+RR(Kb) = 1.0
     */
      rr = RadRate(Z, KA_LINE);
      if (rr == 1.0) {
        throw new IllegalArgumentException(INVALID_LINE);
      }
      else if (rr == 0.0) {
        throw new IllegalArgumentException(INVALID_LINE);
      }
      return 1.0 - rr;
    }
    else if (line == LA_LINE) {
      line = -L3M5_LINE - 1;
      rr = RadRate_arr[Z * LINENUM + line];
      line = -L3M4_LINE - 1;
      rr += RadRate_arr[Z * LINENUM + line];
      if (rr == 0.0) {
        throw new IllegalArgumentException(INVALID_LINE);
      }
      return rr;
    }
    else if (line == LB_LINE) {
      throw new IllegalArgumentException(INVALID_LINE);
    }
    /*
     * in Siegbahn notation: use only KA, KB and LA. The radrates of other lines are nonsense
     */

    line = -line - 1;
    if (line < 0 || line >= LINENUM) {
      throw new IllegalArgumentException(UNKNOWN_LINE);
    }

    rad_rate = RadRate_arr[Z * LINENUM + line];

    if (rad_rate <= 0.) {
      throw new IllegalArgumentException(INVALID_LINE);
    }

    return rad_rate;
  }

  private static double CS_Factory(int Z, double E, int[] NE_arr, double[][] E_arr, double[][] CS_arr, double[][] CS_arr2) {
    double ln_E, ln_sigma, sigma;

    if (Z < 1 || Z > ZMAX || NE_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    ln_E = Math.log(E * 1000.0);

    ln_sigma = splint(E_arr[Z], CS_arr[Z], CS_arr2[Z], NE_arr[Z], ln_E);

    sigma = Math.exp(ln_sigma);

    return sigma;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Photo(int Z, double E) {
    double rv = CS_Factory(Z, E, NE_Photo_arr, E_Photo_arr, CS_Photo_arr, CS_Photo_arr2);
    return rv;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding Rayleigh scattering cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Rayl(int Z, double E) {
    double rv = CS_Factory(Z, E, NE_Rayl_arr, E_Rayl_arr, CS_Rayl_arr, CS_Rayl_arr2);
    return rv;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding Compton scattering cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Compt(int Z, double E) {
    double rv = CS_Factory(Z, E, NE_Compt_arr, E_Compt_arr, CS_Compt_arr, CS_Compt_arr2);
    return rv;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding mass-energy absorption cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Energy(int Z, double E) {
    double rv = CS_Factory(Z, E / 1000.0, NE_Energy_arr, E_Energy_arr, CS_Energy_arr, CS_Energy_arr2);
    return rv;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding total attenuation cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Total(int Z, double E) {
    if (Z<1 || Z>ZMAX || NE_Photo_arr[Z] < 0 || NE_Rayl_arr[Z] < 0 || NE_Compt_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    return CS_Photo(Z, E) + CS_Rayl(Z, E) + CS_Compt(Z, E);
  }

  /** 
   * For a given atomic number and momentum transfer, returns the corresponding atomic form factor for Rayleigh scattering.
   *
   * @param Z The atomic number
   * @param q The momentum transfer, expressed in Å<sup>-1</sup>
   * @return The atomic form factor
   */
  public static double FF_Rayl(int Z, double q) {
    double FF;

    if (Z < 1 || Z > ZMAX || Nq_Rayl_arr[Z] <= 0.0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (q == 0)
      return Z;

    if (q < 0.0) {
      throw new IllegalArgumentException(NEGATIVE_Q);
    }

    FF = splint(q_Rayl_arr[Z], FF_Rayl_arr[Z], FF_Rayl_arr2[Z], Nq_Rayl_arr[Z], q);

    return FF;
  }


  /** 
   * For a given atomic number and momentum transfer, returns the corresponding incoherent scattering function for Compton scattering.
   *
   * @param Z The atomic number
   * @param q The momentum transfer, expressed in Å<sup>-1</sup>
   * @return The incoherent scattering function
   */
  public static double SF_Compt(int Z, double q) {
    double SF;

    if (Z<1 || Z>ZMAX || Nq_Compt_arr[Z] <= 0.0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (q <= 0.) {
      throw new IllegalArgumentException(NEGATIVE_Q);
    }

    SF = splint(q_Compt_arr[Z], SF_Compt_arr[Z], SF_Compt_arr2[Z], Nq_Compt_arr[Z], q);

    return SF;
  }

  /** 
   * For a given scattering angle, returns the @see <a href="https://en.wikipedia.org/wiki/Thomson_scattering">Thomson differential cross section</a>
   * 
   * @param theta The scattering angle, between indicent and observed photon or wave.
   * @return The Thomson differential cross section, expressed in barn
   */
  public static double DCS_Thoms(double theta) {
    double cos_theta;

    cos_theta = Math.cos(theta);

    return (RE2/2.0) * (1.0 + cos_theta * cos_theta);
  }

  /** 
   * For a given energy and scattering angle, returns the @see <a href="https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula">Klein-Nishina differential cross section</a>
   * 
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Klein-Nishina differential cross section, expressed in barn
   */
  public static double DCS_KN(double E, double theta) {
    double cos_theta, t1, t2;

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    cos_theta = Math.cos(theta);
    t1 = (1.0 - cos_theta) * E / MEC2 ;
    t2 = 1.0 + t1;

    return (RE2/2.) * (1.0 + cos_theta * cos_theta + t1 *t1 / t2) /t2 /t2;
  }

  /** 
   * For a given atomic number, energy and scattering angle, returns the Rayleigh differential cross section.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in cm<sup>2</sup>/g/sterad
   */
  public static double DCS_Rayl(int Z, double E, double theta) {
    double F, q ;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    q = MomentTransf(E, theta);
    F = FF_Rayl(Z, q);

    return  AVOGNUM / AtomicWeight_arr[Z] * F * F * DCS_Thoms(theta);
  }

  /** 
   * For a given atomic number, energy and scattering angle, returns the Compton differential cross section.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in cm<sup>2</sup>/g/sterad
   */
  public static double DCS_Compt(int Z, double E, double theta) {
    double S, q ;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    q = MomentTransf(E, theta);
    S = SF_Compt(Z, q);

    return  AVOGNUM / AtomicWeight_arr[Z] * S * DCS_KN(E, theta);
  }

  /** 
   * For a given energy and scattering angle, returns the @see <a href="https://en.wikipedia.org/wiki/Momentum_transfer">momentum transfer</a>
   * 
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The momentum transfer for X-ray photon scattering, expressed in Å<sup>-1</sup>
   */
  public static double MomentTransf(double E, double theta) {
    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    return E / KEV2ANGST * Math.sin(theta / 2.0) ;
  }

  /** 
   * For a given energy, returns the @see <a href="https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula">Klein-Nishina cross section</a>
   * 
   * @param E The photon energy, expressed in keV
   * @return The Klein-Nishina cross section, expressed in barn
   */
  public static double CS_KN(double E) {
    double a, a3, b, b2, lb;
    double sigma;

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    a = E / MEC2;
    a3 = a * a * a;
    b = 1 + 2 * a;
    b2 = b * b;
    lb = Math.log(b);

    sigma = 2 * Math.PI * RE2*( (1 + a) / a3 * ( 2 * a * (1 + a) / b - lb) + 0.5 * lb / a - (1 + 3 * a) / b2);
    return sigma;
  }

  /** 
   * For a given energy and scattering angle, returns the photon energy after @see <a href="https://en.wikipedia.org/wiki/Compton_scattering#Derivation_of_the_scattering_formula">Compton scattering</a>.
   * 
   * @param E0 The initial photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The photon energy after scattering, expressed in keV.
   */
  public static double ComptonEnergy(double E0, double theta) {
    double cos_theta, alpha;

    if (E0 <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    cos_theta = Math.cos(theta);
    alpha = E0 / MEC2;

    return E0 / (1 + alpha * (1 - cos_theta));
  }

  /** 
   * For a given atomic number and energy, returns the corresponding photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Photo_Total(int Z, double E) {
    int shell;
    double rv = 0.0;

    if (Z < 1 || Z > ZMAX || NE_Photo_Total_Kissel_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    for (shell = K_SHELL ; shell <= Q3_SHELL ; shell++) {
      if (Electron_Config_Kissel_arr[Z * SHELLNUM_K + shell] > 1.0E-06) {
	try {
          rv += CSb_Photo_Partial(Z, shell, E) * Electron_Config_Kissel_arr[Z * SHELLNUM_K + shell];
	} catch (IllegalArgumentException e) {
	}
      }
    }

    if (rv <= 0.0) {
      throw new IllegalArgumentException(UNAVAILABLE_PHOTO_CS);
    }

    return rv;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Photo_Total(int Z, double E) {
    return CSb_Photo_Total(Z, E) * AVOGNUM / AtomicWeight_arr[Z];
  }

  /** 
   * For a given atomic number, shell and energy, returns the corresponding partial photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/electron.
   */
  public static double CSb_Photo_Partial(int Z, int shell, double E) {
    double ln_E, ln_sigma, sigma;
    double x0, x1, y0, y1;
    double m;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell < 0 || shell >= SHELLNUM_K) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    if (Electron_Config_Kissel_arr[Z * SHELLNUM_K + shell] < 1.0E-06 || EdgeEnergy_arr[Z * SHELLNUM + shell] <= 0.0){
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    if (EdgeEnergy_arr[Z * SHELLNUM + shell] > E) {
      throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
    }

    ln_E = Math.log(E);
    if (ln_E < E_Photo_Partial_Kissel_arr[Z][shell][0]) {
	  /* Address a case where energy E is less than the lowest value in the energies array of Kissel's cross section
         Fixes https://github.com/tschoonj/xraylib/issues/187 
      */
      /*
       * use log-log extrapolation
       */
      x0 = E_Photo_Partial_Kissel_arr[Z][shell][0];
      x1 = E_Photo_Partial_Kissel_arr[Z][shell][1];
      y0 = Photo_Partial_Kissel_arr[Z][shell][0];
      y1 = Photo_Partial_Kissel_arr[Z][shell][1];
      /*
       * do not allow "extreme" slopes... force them to be within -1;1
       */
      m = (y1-y0)/(x1-x0);
      if (m > 1.0)
        m=1.0;
      else if (m < -1.0)
        m=-1.0;
      ln_sigma = y0+m*(ln_E-x0);
    }
    else {
      ln_sigma = splint(E_Photo_Partial_Kissel_arr[Z][shell], Photo_Partial_Kissel_arr[Z][shell], Photo_Partial_Kissel_arr2[Z][shell],NE_Photo_Partial_Kissel_arr[Z][shell], ln_E);
    }
    sigma = Math.exp(ln_sigma);
    return sigma;
  }

  /** 
   * For a given atomic number, shell and energy, returns the corresponding partial photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Photo_Partial(int Z, int shell, double E) {
    return CSb_Photo_Partial(Z, shell, E) * Electron_Config_Kissel_arr[Z * SHELLNUM_K + shell] * AVOGNUM/AtomicWeight_arr[Z];
  }

  /** 
   * For a given atomic number and energy, returns the corresponding total attenuation cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Total_Kissel(int Z, double E) {

    if (Z < 1 || Z > ZMAX || NE_Photo_Total_Kissel_arr[Z] < 0 || NE_Rayl_arr[Z] < 0 || NE_Compt_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    return CS_Photo_Total(Z, E) + CS_Rayl(Z, E) + CS_Compt(Z, E);
  }

  /** 
   * For a given atomic number and energy, returns the corresponding total attenuation cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Total_Kissel(int Z, double E) {
    return CS_Total_Kissel(Z, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding @see <a href="https://en.wikipedia.org/wiki/Electron_configuration">electron configuration</a>.
   *
   * This method used the Kissel database to determine the electron configuration.
   *
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The number of electrons that occupy the shell
   */
  public static double ElectronConfig(int Z, int shell) {
    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell < 0 || shell >= SHELLNUM_K) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    double rv = Electron_Config_Kissel_arr[Z * SHELLNUM_K + shell];

    if (rv == 0.0) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return rv;
  }

  /** 
   * For a given atomic number and momentum, returns the corresponding Compton scattering profile, summed over all shells.
   *
   * @param Z The atomic number
   * @param pz The momentum
   * @return The Compton scattering profile
   */
  public static double ComptonProfile(int Z, double pz) {
    double q, ln_q;
    double ln_pz;

    if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (pz < 0.0) {
      throw new IllegalArgumentException(NEGATIVE_PZ);
    }

    ln_pz = Math.log(pz + 1.0);

    ln_q = splint(pz_ComptonProfiles_arr[Z], Total_ComptonProfiles_arr[Z], Total_ComptonProfiles_arr2[Z],  Npz_ComptonProfiles_arr[Z], ln_pz);

    q = Math.exp(ln_q);

    return q;
  }

  /** 
   * For a given atomic number, shell and momentum, returns the corresponding Compton scattering profile.
   *
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @param pz The momentum
   * @return The Compton scattering profile
   */
  public static double ComptonProfile_Partial(int Z, int shell, double pz) {
    double q, ln_q;
    double ln_pz;

    if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles_arr[Z] < 1) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell >= NShells_ComptonProfiles_arr[Z] || shell < K_SHELL || UOCCUP_ComptonProfiles_arr[Z][shell] == 0.0 ) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    if (pz < 0.0) {
      throw new IllegalArgumentException(NEGATIVE_PZ);
    }

    ln_pz = Math.log(pz + 1.0);

    ln_q = splint(pz_ComptonProfiles_arr[Z], Partial_ComptonProfiles_arr[Z][shell], Partial_ComptonProfiles_arr2[Z][shell], Npz_ComptonProfiles_arr[Z], ln_pz);

    q = Math.exp(ln_q);

    return q;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding @see <a href="https://en.wikipedia.org/wiki/Electron_configuration">electron configuration</a>.
   *
   * This method used the Biggs database to determine the electron configuration.
   *
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The number of electrons that occupy the shell
   */
  public static double ElectronConfig_Biggs(int Z, int shell) {
    if (Z < 1 || Z > ZMAX || NShells_ComptonProfiles_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (shell >= NShells_ComptonProfiles_arr[Z] || UOCCUP_ComptonProfiles_arr[Z][shell] == 0.0 ) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return UOCCUP_ComptonProfiles_arr[Z][shell];
  }

  /** 
   * For a given atomic number and Auger transition, returns the corresponding non-radiative Auger rate.
   *
   * Transitions that correspond to Coster-Kronig transitions (such as #L2_L3N6_AUGER) will throw an exception.
   *
   * The returned value will be between 0 and 1.
   * 
   * @param Z The atomic number
   * @param auger_trans A macro identifying the transition, such as #K_L2O7_AUGER
   * @return The Auger radiative rate (dimensionless)
   */
  public static double AugerRate(int Z, int auger_trans) {
    double rv;

    rv = 0.0;

    if (Z > ZMAX || Z < 1) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }
    else if (auger_trans < K_L1L1_AUGER || auger_trans > M4_M5Q3_AUGER) {
      throw new IllegalArgumentException(UNKNOWN_AUGER);
    }

    rv = Auger_Rates_arr[Z * AUGERNUM + auger_trans];

    if (rv <= 0.) {
      throw new IllegalArgumentException(INVALID_AUGER);
    }

    return rv;
  }

  /** 
   * For a given atomic number and shell, returns the corresponding non-radiative Auger yield.
   * 
   * The returned values does not cover Coster-Kronig transitions! Use #CosKronTransProb to obtain those values.
   *
   * The returned value will be between 0 and 1.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL
   * @return The non-radiative Auger yield (dimensionless)
   */
  public static double AugerYield(int Z, int shell) {
    double rv;

    if (Z > ZMAX || Z < 1) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }
    else if (shell < K_SHELL || shell > M5_SHELL) {
      throw new IllegalArgumentException(UNKNOWN_SHELL);
    }

    rv = Auger_Yields_arr[Z * SHELLNUM_A + shell];

    if (rv == 0.0) {
      throw new IllegalArgumentException(INVALID_SHELL);
    }

    return rv;
  }

  private static double RadRate_catch(int Z, int line) {
    try {
      return RadRate(Z, line);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double CS_FluorLine_catch(int Z, int line, double E) {
    try {
      return CS_FluorLine(Z, line, E);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double EdgeEnergy_catch(int Z, int shell) {
    try {
      return EdgeEnergy(Z, shell);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double FluorYield_catch(int Z, int shell) {
    try {
      return FluorYield(Z, shell);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double JumpFactor_catch(int Z, int shell) {
    try {
      return JumpFactor(Z, shell);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double LineEnergy_catch(int Z, int line) {
    try {
      return LineEnergy(Z, line);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double CS_Photo_Partial_catch(int Z, int shell, double E) {
    try {
      return CS_Photo_Partial(Z, shell, E);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static double CosKronTransProb_catch(int Z, int trans) {
    try {
      return CosKronTransProb(Z, trans);
    }
    catch (IllegalArgumentException e) {
      return 0.0;
    }
  }

  private static int get_kissel_offset(int n1, int n2, int n3) {
    return (M5_SHELL + 1) * (L3_SHELL + 1) * n1 + (L3_SHELL + 1) * n2 + n3;
  }

  public static double PL1_pure_kissel(int Z, double E) {
    return CS_Photo_Partial(Z, L1_SHELL, E);
  }

  public static double PL1_rad_cascade_kissel(int Z, double E, double PK) {
    double rv;

    rv = CS_Photo_Partial(Z,L1_SHELL, E);

    if (PK > 0.0) {
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KL1_LINE);
    }

    return rv;
  }

  public static double PL1_auger_cascade_kissel(int Z, double E, double PK) {
    double rv;

    rv = CS_Photo_Partial(Z, L1_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, L1_SHELL, K_SHELL)];

    return rv;
  }

  public static double PL1_full_cascade_kissel(int Z, double E, double PK) {
    double rv;

    rv = CS_Photo_Partial(Z, L1_SHELL, E);
    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, L1_SHELL, K_SHELL)];

    return rv;
  }

  public static double PL2_pure_kissel(int Z, double E, double PL1) {
    double rv;

    rv = CS_Photo_Partial(Z, L2_SHELL, E);

    if (PL1 > 0.0)
      rv += CosKronTransProb_catch(Z, FL12_TRANS) * PL1;

    return rv;
  }

  public static double PL2_rad_cascade_kissel(int Z, double E, double PK, double PL1) {
    double rv;

    rv = CS_Photo_Partial(Z, L2_SHELL, E);

    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KL2_LINE);

    if (PL1 > 0.0)
      rv += CosKronTransProb_catch(Z, FL12_TRANS) * PL1;

    return  rv;
  }

  public static double PL2_auger_cascade_kissel(int Z, double E, double PK, double PL1) {
    double rv;

    rv = CS_Photo_Partial(Z, L2_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, L2_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += CosKronTransProb_catch(Z, FL12_TRANS) * PL1;

    return  rv;
  }

  public static double PL2_full_cascade_kissel(int Z, double E, double PK, double PL1) {
    double rv;

    rv = CS_Photo_Partial(Z,L2_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, L2_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += CosKronTransProb_catch(Z, FL12_TRANS) * PL1;

    return rv;
  }

  public static double PL3_pure_kissel(int Z, double E, double PL1, double PL2) {
    double rv;

    rv = CS_Photo_Partial(Z, L3_SHELL, E);

    if (PL1 > 0.0)
      rv += (CosKronTransProb_catch(Z, FL13_TRANS) + CosKronTransProb_catch(Z, FLP13_TRANS)) * PL1;

    if (PL2 > 0.0)
      rv += CosKronTransProb_catch(Z, FL23_TRANS) * PL2;


    return rv;
}

  public static double PL3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
    double rv;

    rv = CS_Photo_Partial(Z, L3_SHELL, E);

    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KL3_LINE);

    if (PL1 > 0.0)
      rv += (CosKronTransProb_catch(Z, FL13_TRANS) + CosKronTransProb_catch(Z, FLP13_TRANS))* PL1;

    if (PL2 > 0.0)
      rv += CosKronTransProb_catch(Z, FL23_TRANS) * PL2;

    return  rv;
  }

  public static double PL3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
    double rv;

    rv = CS_Photo_Partial(Z,L3_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, L3_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += (CosKronTransProb_catch(Z, FL13_TRANS) + CosKronTransProb_catch(Z, FLP13_TRANS))* PL1;

    if (PL2 > 0.0)
      rv += CosKronTransProb_catch(Z, FL23_TRANS) * PL2;

    return  rv;
  }

  public static double PL3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
    double rv;

    rv = CS_Photo_Partial(Z,L3_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, L3_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += CosKronTransProb_catch(Z, FL13_TRANS) * PL1;

    if (PL2 > 0.0)
      rv += CosKronTransProb_catch(Z, FL23_TRANS) * PL2;

    return rv;
  }

  public static double PM1_pure_kissel(int Z, double E) {
    return CS_Photo_Partial(Z, M1_SHELL, E);
  }

  public static double PM1_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
    double rv;

    rv = CS_Photo_Partial(Z, M1_SHELL, E);

    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KM1_LINE);
    if (PL1 > 0.0)
      rv += FluorYield_catch(Z, L1_SHELL) * PL1 * RadRate_catch(Z, L1M1_LINE);
    if (PL2 > 0.0)
      rv += FluorYield_catch(Z, L2_SHELL) * PL2 * RadRate_catch(Z, L2M1_LINE);
    if (PL3 > 0.0)
      rv += FluorYield_catch(Z, L3_SHELL) * PL3 * RadRate_catch(Z, L3M1_LINE);

    return rv;
  }

  public static double PM1_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
    double rv;

    rv = CS_Photo_Partial(Z, M1_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M1_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M1_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M1_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M1_SHELL, L3_SHELL)];

    return rv;
  }

  public static double PM1_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
    double rv;

    rv = CS_Photo_Partial(Z, M1_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, M1_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M1_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M1_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M1_SHELL, L3_SHELL)];

    return rv;
  }


  public static double PM2_pure_kissel(int Z, double E, double PM1) {
    double rv;

    rv = CS_Photo_Partial(Z, M2_SHELL, E);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM12_TRANS) * PM1;

    return rv;
  }

  public static double PM2_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
    double rv;

    rv = CS_Photo_Partial(Z, M2_SHELL, E);

    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KM2_LINE);

    if (PL1 > 0.0)
      rv += FluorYield_catch(Z, L1_SHELL) * PL1 * RadRate_catch(Z, L1M2_LINE);

    if (PL2 > 0.0)
      rv += FluorYield_catch(Z, L2_SHELL) * PL2 * RadRate_catch(Z, L2M2_LINE);

    if (PL3 > 0.0)
      rv += FluorYield_catch(Z, L3_SHELL) * PL3 * RadRate_catch(Z, L3M2_LINE);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM12_TRANS) * PM1;

    return rv;
  }

  public static double PM2_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
    double rv;

    rv = CS_Photo_Partial(Z, M2_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M2_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M2_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M2_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M2_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM12_TRANS) * PM1;

    return rv;
  }

  public static double PM2_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
    double rv;

    rv = CS_Photo_Partial(Z, M2_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, M2_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M2_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M2_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M2_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM12_TRANS) * PM1;

    return rv;
  }

  public static double PM3_pure_kissel(int Z, double E, double PM1, double PM2) {
    double rv;

    rv = CS_Photo_Partial(Z, M3_SHELL, E);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM13_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM23_TRANS) * PM2;

    return rv;
  }

  public static double PM3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
    double rv;

    rv = CS_Photo_Partial(Z, M3_SHELL, E);

    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KM3_LINE);

    if (PL1 > 0.0)
      rv += FluorYield_catch(Z, L1_SHELL) * PL1 * RadRate_catch(Z, L1M3_LINE);

    if (PL2 > 0.0)
      rv += FluorYield_catch(Z, L2_SHELL) * PL2 * RadRate_catch(Z, L2M3_LINE);

    if (PL3 > 0.0)
      rv += FluorYield_catch(Z, L3_SHELL) * PL3 * RadRate_catch(Z, L3M3_LINE);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM13_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM23_TRANS) * PM2;

    return rv;
  }

  public static double PM3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
    double rv;

    rv = CS_Photo_Partial(Z, M3_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M3_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M3_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M3_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M3_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM13_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM23_TRANS) * PM2;

    return rv;
  }

  public static double PM3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
    double rv;

    rv = CS_Photo_Partial(Z, M3_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, M3_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M3_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M3_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M3_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM13_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM23_TRANS) * PM2;

    return rv;
  }

  public static double PM4_pure_kissel(int Z, double E, double PM1, double PM2, double PM3) {
    double rv;

    rv = CS_Photo_Partial(Z, M4_SHELL, E);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM14_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM24_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM34_TRANS) * PM3;

    return rv;
  }

  public static double PM4_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
    double rv;

    rv = CS_Photo_Partial(Z, M4_SHELL, E);

    /*yes I know that KM4 lines are forbidden... */
    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KM4_LINE);

    if (PL1 > 0.0)
      rv += FluorYield_catch(Z, L1_SHELL) * PL1 * RadRate_catch(Z, L1M4_LINE);

    if (PL2 > 0.0)
      rv += FluorYield_catch(Z, L2_SHELL) * PL2 * RadRate_catch(Z, L2M4_LINE);

    if (PL3 > 0.0)
      rv += FluorYield_catch(Z, L3_SHELL) * PL3 * RadRate_catch(Z, L3M4_LINE);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM14_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM24_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM34_TRANS) * PM3;

    return rv;

  }

  public static double PM4_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
    double rv;

    rv = CS_Photo_Partial(Z, M4_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M4_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M4_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M4_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M4_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM14_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM24_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM34_TRANS) * PM3;

    return rv;
  }

  public static double PM4_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
    double rv;

    rv = CS_Photo_Partial(Z, M4_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, M4_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M4_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M4_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M4_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM14_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM24_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM34_TRANS) * PM3;

    return rv;
  }

  public static double PM5_pure_kissel(int Z, double E, double PM1, double PM2, double PM3, double PM4) {
    double rv;

    rv = CS_Photo_Partial(Z, M5_SHELL, E);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM15_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM25_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM35_TRANS) * PM3;

    if (PM4 > 0.0)
      rv += CosKronTransProb_catch(Z, FM45_TRANS) * PM4;

    return rv;
  }

  public static double PM5_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
    double rv;

    rv = CS_Photo_Partial(Z, M5_SHELL, E);

    /*yes I know that KM5 lines are forbidden... */
    if (PK > 0.0)
      rv += FluorYield_catch(Z, K_SHELL) * PK * RadRate_catch(Z, KM5_LINE);

    if (PL1 > 0.0)
      rv += FluorYield_catch(Z, L1_SHELL) * PL1 * RadRate_catch(Z, L1M5_LINE);

    if (PL2 > 0.0)
      rv += FluorYield_catch(Z, L2_SHELL) * PL2 * RadRate_catch(Z, L2M5_LINE);

    if (PL3 > 0.0)
      rv += FluorYield_catch(Z, L3_SHELL) * PL3 * RadRate_catch(Z, L3M5_LINE);

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM15_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM25_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM35_TRANS) * PM3;

    if (PM4 > 0.0)
      rv += CosKronTransProb_catch(Z, FM45_TRANS) * PM4;

    return rv;
  }

  public static double PM5_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
    double rv;

    rv = CS_Photo_Partial(Z, M5_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M5_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M5_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M5_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_auger_only[get_kissel_offset(Z, M5_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM15_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM25_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM35_TRANS) * PM3;

    if (PM4 > 0.0)
      rv += CosKronTransProb_catch(Z, FM45_TRANS) * PM4;

    return rv;
  }

  public static double PM5_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
    double rv;

    rv = CS_Photo_Partial(Z, M5_SHELL, E);

    if (PK > 0.0)
      rv += PK * xrf_cross_sections_constants_full[get_kissel_offset(Z, M5_SHELL, K_SHELL)];

    if (PL1 > 0.0)
      rv += PL1 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M5_SHELL, L1_SHELL)];

    if (PL2 > 0.0)
      rv += PL2 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M5_SHELL, L2_SHELL)];

    if (PL3 > 0.0)
      rv += PL3 * xrf_cross_sections_constants_full[get_kissel_offset(Z, M5_SHELL, L3_SHELL)];

    if (PM1 > 0.0)
      rv += CosKronTransProb_catch(Z, FM15_TRANS) * PM1;

    if (PM2 > 0.0)
      rv += CosKronTransProb_catch(Z, FM25_TRANS) * PM2;

    if (PM3 > 0.0)
      rv += CosKronTransProb_catch(Z, FM35_TRANS) * PM3;

    if (PM4 > 0.0)
      rv += CosKronTransProb_catch(Z, FM45_TRANS) * PM4;

    return rv;
  }

  /** 
   * For a given atomic number, line, and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method is an alias for #CS_FluorLine_Kissel_Cascade, meaning that the cascade effect will be taken into account.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine_Kissel(int Z, int line, double E) {
    return CS_FluorLine_Kissel_Cascade(Z, line, E);
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method is an alias for #CSb_FluorLine_Kissel_Cascade, meaning that the cascade effect will be taken into account.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine_Kissel(int Z, int line, double E) {
    return CSb_FluorLine_Kissel_Cascade(Z, line, E);
  }

  private static class LineMapping {

    public final int line_lower;
    public final int line_upper;
    public final int shell;

    public LineMapping(final int line_lower, final int line_upper, final int shell) {
      this.line_lower = line_lower;
      this.line_upper = line_upper;
      this.shell = shell;
    }
  }

  private static final LineMapping[] line_mappings = new LineMapping[]{
    new LineMapping(Xraylib.KP5_LINE, Xraylib.KB_LINE, Xraylib.K_SHELL),
    new LineMapping(Xraylib.L1P5_LINE, Xraylib.L1L2_LINE, Xraylib.L1_SHELL),
    new LineMapping(Xraylib.L2Q1_LINE, Xraylib.L2L3_LINE, Xraylib.L2_SHELL),
    new LineMapping(Xraylib.L3Q1_LINE, Xraylib.L3M1_LINE, Xraylib.L3_SHELL),
    new LineMapping(Xraylib.M1P5_LINE, Xraylib.M1N1_LINE, Xraylib.M1_SHELL),
    new LineMapping(Xraylib.M2P5_LINE, Xraylib.M2N1_LINE, Xraylib.M2_SHELL),
    new LineMapping(Xraylib.M3Q1_LINE, Xraylib.M3N1_LINE, Xraylib.M3_SHELL),
    new LineMapping(Xraylib.M4P5_LINE, Xraylib.M4N1_LINE, Xraylib.M4_SHELL),
    new LineMapping(Xraylib.M5P5_LINE, Xraylib.M5N1_LINE, Xraylib.M5_SHELL)
  };

  @FunctionalInterface
  private static interface CS_FluorLine_Kissel_Body {
    public default double execute(int Z, int line, double energy) {
      if (Z < 1 || Z > ZMAX)
        throw new IllegalArgumentException(Z_OUT_OF_RANGE);

      if (energy <= 0.0)
        throw new IllegalArgumentException(NEGATIVE_ENERGY);

      for (LineMapping mapping: line_mappings) {
        if (line >= mapping.line_lower && line <= mapping.line_upper) {
          double rr = RadRate(Z, line);
          double Factor = CS_FluorShell_Kissel_base(Z, mapping.shell, energy);
          return rr * Factor;
        }
      }

      /* special cases */
      if (line == LA_LINE) {
        double rr = RadRate(Z, line);
        double Factor = CS_FluorShell_Kissel_base(Z, L3_SHELL, energy);
        return rr * Factor;
      } else if (line == LB_LINE) {
        double rv = 0.0;

        for (LineShellPair pair : lb_pairs) {
          try {
            rv += RadRate(Z, pair.line) * CS_FluorShell_Kissel_base(Z, pair.shell, energy);
          } catch (Exception e) {
            continue;
          }
        }
        if (rv == 0.0)
          throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
        return rv;
      }

      throw new IllegalArgumentException(INVALID_LINE);
    }

    public double CS_FluorShell_Kissel_base(int Z, int shell, double energy);
  }

  private static final CS_FluorLine_Kissel_Body CS_FLUORLINE_KISSEL_NO_CASCADE = Xraylib::CS_FluorShell_Kissel_no_Cascade;
  private static final CS_FluorLine_Kissel_Body CS_FLUORLINE_KISSEL_RADIATIVE = Xraylib::CS_FluorShell_Kissel_Radiative_Cascade;
  private static final CS_FluorLine_Kissel_Body CS_FLUORLINE_KISSEL_NONRADIATIVE = Xraylib::CS_FluorShell_Kissel_Nonradiative_Cascade;
  private static final CS_FluorLine_Kissel_Body CS_FLUORLINE_KISSEL_FULL = Xraylib::CS_FluorShell_Kissel_Cascade;

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method excludes the cascade effect from the calculation!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine_Kissel_no_Cascade(int Z, int line, double E) {
    return CS_FLUORLINE_KISSEL_NO_CASCADE.execute(Z, line, E);
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the radiative cascade contributions but excludes the non-radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E) {
    return CS_FLUORLINE_KISSEL_RADIATIVE.execute(Z, line, E);
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the non-radiative cascade contributions but excludes the radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E) {
    return CS_FLUORLINE_KISSEL_NONRADIATIVE.execute(Z, line, E);
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes both non-radiative and radiative cascade effect contributions!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine_Kissel_Cascade(int Z, int line, double E) {
    return CS_FLUORLINE_KISSEL_FULL.execute(Z, line, E);
  }
  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method excludes the cascade effect from the calculation!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell_Kissel_no_Cascade(int Z, int shell, double E) {
    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    if (shell == K_SHELL) {
      /*
       * K lines -> never cascade effect!
       */
      return FluorYield(Z, K_SHELL) * CS_Photo_Partial(Z, K_SHELL, E);
    }
    else if (shell == L1_SHELL) {
      /*
       * L1 lines
       */
      return FluorYield(Z, L1_SHELL) * PL1_pure_kissel(Z, E);
    }
    else if (shell == L2_SHELL) {
      /*
       * L2 lines
       */
      double PL1 = 0.0;
      try {
        PL1 = PL1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, L2_SHELL) * PL2_pure_kissel(Z, E, PL1);
    }
    else if (shell == L3_SHELL) {
      /*
       * L3 lines
       */
      double PL1 = 0.0;
      double PL2 = 0.0;
      try {
        PL1 = PL1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      try {
        PL2 = PL2_pure_kissel(Z, E, PL1);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, L3_SHELL) * PL3_pure_kissel(Z, E, PL1, PL2);
    }
    else if (shell == M1_SHELL) {
      /*
       * M1 lines
       */
      return FluorYield(Z, M1_SHELL) * PM1_pure_kissel(Z, E);
    }
    else if (shell == M2_SHELL) {
      /*
       * M2 lines
       */
      double PM1 = 0.0;
      try {
        PM1 = PM1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, M2_SHELL) * PM2_pure_kissel(Z, E, PM1);
    }
    else if (shell == M3_SHELL) {
      /*
       * M3 lines
       */
      double PM1 = 0.0;
      double PM2 = 0.0;
      try {
        PM1 = PM1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM2 = PM2_pure_kissel(Z, E, PM1);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, M3_SHELL) * PM3_pure_kissel(Z, E, PM1, PM2);
    }
    else if (shell == M4_SHELL) {
      /*
       * M4 lines
       */
      double PM1 = 0.0;
      double PM2 = 0.0;
      double PM3 = 0.0;
      try {
        PM1 = PM1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM2 = PM2_pure_kissel(Z, E, PM1);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM3 = PM3_pure_kissel(Z, E, PM1, PM2);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, M4_SHELL) * PM4_pure_kissel(Z, E, PM1, PM2, PM3);
    }
    else if (shell == M5_SHELL) {
      /*
       * M5 lines
       */
      double PM1 = 0.0;
      double PM2 = 0.0;
      double PM3 = 0.0;
      double PM4 = 0.0;
      try {
        PM1 = PM1_pure_kissel(Z, E);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM2 = PM2_pure_kissel(Z, E, PM1);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM3 = PM3_pure_kissel(Z, E, PM1, PM2);
      } catch (IllegalArgumentException e) {
      }
      try {
        PM4 = PM4_pure_kissel(Z, E, PM1, PM2, PM3);
      } catch (IllegalArgumentException e) {
      }
      return FluorYield(Z, M5_SHELL) * PM5_pure_kissel(Z, E, PM1, PM2, PM3, PM4);
    }
    else {
      throw new IllegalArgumentException(INVALID_SHELL);
    }
  }

  private static abstract class CS_FluorShell_Cascade_Body {

    public final double execute(int Z, int shell, double E) {
      if (Z < 1 || Z > ZMAX) {
        throw new IllegalArgumentException(Z_OUT_OF_RANGE);
      }

      if (E <= 0.0) {
        throw new IllegalArgumentException(NEGATIVE_ENERGY);
      }

      if (shell == K_SHELL) {
        /*
         * K lines -> never cascade effect!
         */
        return FluorYield(Z, shell) * CS_Photo_Partial(Z, K_SHELL, E);
      }
      else if (shell == L1_SHELL) {
        /*
         * L1 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        return FluorYield(Z, shell)* PL1_cascade_kissel(Z, E, PK);
      }
      else if (shell == L2_SHELL) {
        /*
         * L2 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        return FluorYield(Z, shell) * PL2_cascade_kissel(Z, E, PK, PL1);
      }
      else if (shell == L3_SHELL) {
        /*
         * L3 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        return FluorYield(Z, shell) * PL3_cascade_kissel(Z, E, PK, PL1, PL2);
      }
      else if (shell == M1_SHELL) {
        /*
         * M1 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        double PL3 = PL3_cascade_kissel_catch(Z, E, PK, PL1, PL2);
        return FluorYield(Z, shell) * PM1_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
      }
      else if (shell == M2_SHELL) {
        /*
         * M2 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        double PL3 = PL3_cascade_kissel_catch(Z, E, PK, PL1, PL2);
        double PM1 = PM1_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3);
        return FluorYield(Z, shell) * PM2_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
      }
      else if (shell == M3_SHELL) {
        /*
         * M3 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        double PL3 = PL3_cascade_kissel_catch(Z, E, PK, PL1, PL2);
        double PM1 = PM1_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3);
        double PM2 = PM2_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1);
        return FluorYield(Z, shell) * PM3_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
      }
      else if (shell == M4_SHELL) {
        /*
         * M4 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        double PL3 = PL3_cascade_kissel_catch(Z, E, PK, PL1, PL2);
        double PM1 = PM1_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3);
        double PM2 = PM2_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1);
        double PM3 = PM3_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
        return FluorYield(Z, shell) * PM4_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
      }
      else if (shell == M5_SHELL) {
        /*
         * M5 lines
         */
        double PK = CS_Photo_Partial_catch(Z, K_SHELL, E);
        double PL1 = PL1_cascade_kissel_catch(Z, E, PK);
        double PL2 = PL2_cascade_kissel_catch(Z, E, PK, PL1);
        double PL3 = PL3_cascade_kissel_catch(Z, E, PK, PL1, PL2);
        double PM1 = PM1_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3);
        double PM2 = PM2_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1);
        double PM3 = PM3_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
        double PM4 = PM4_cascade_kissel_catch(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
        return FluorYield(Z, shell) * PM5_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
      }
      else {
        throw new IllegalArgumentException(INVALID_SHELL);
      }
    }
  	
    public abstract double PL1_cascade_kissel(int Z, double E, double PK);
    public abstract double PL2_cascade_kissel(int Z, double E, double PK, double PL1);
    public abstract double PL3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2);
    public abstract double PM1_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3);
    public abstract double PM2_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
    public abstract double PM3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
    public abstract double PM4_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
    public abstract double PM5_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);

    public double PL1_cascade_kissel_catch(int Z, double E, double PK) {
      try {
        return PL1_cascade_kissel(Z, E, PK);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PL2_cascade_kissel_catch(int Z, double E, double PK, double PL1) {
      try {
        return PL2_cascade_kissel(Z, E, PK, PL1);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PL3_cascade_kissel_catch(int Z, double E, double PK, double PL1, double PL2) {
      try {
        return PL3_cascade_kissel(Z, E, PK, PL1, PL2);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PM1_cascade_kissel_catch(int Z, double E, double PK, double PL1, double PL2, double PL3) {
      try {
        return PM1_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PM2_cascade_kissel_catch(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
      try {
        return PM2_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PM3_cascade_kissel_catch(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
      try {
        return PM3_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
    public double PM4_cascade_kissel_catch(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
      try {
        return PM4_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
      } catch (IllegalArgumentException e) {
        return 0.0;
      }
    }
  }

  private static final class CS_FluorShell_Kissel_Radiative_CascadeImpl extends CS_FluorShell_Cascade_Body {
  
    public double PL1_cascade_kissel(int Z, double E, double PK) {
      return PL1_rad_cascade_kissel(Z, E, PK);
    }

    public double PL2_cascade_kissel(int Z, double E, double PK, double PL1) {
      return PL2_rad_cascade_kissel(Z, E, PK, PL1);
    }

    public double PL3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
      return PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    }

    public double PM1_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
      return PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    }

    public double PM2_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
      return PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    }

    public double PM3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
      return PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    }

    public double PM4_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
      return PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    }

    public double PM5_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
      return PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
    }
  }

  private static final class CS_FluorShell_Kissel_Nonradiative_CascadeImpl extends CS_FluorShell_Cascade_Body {
  
    public double PL1_cascade_kissel(int Z, double E, double PK) {
      return PL1_auger_cascade_kissel(Z, E, PK);
    }

    public double PL2_cascade_kissel(int Z, double E, double PK, double PL1) {
      return PL2_auger_cascade_kissel(Z, E, PK, PL1);
    }

    public double PL3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
      return PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    }

    public double PM1_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
      return PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    }

    public double PM2_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
      return PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    }

    public double PM3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
      return PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    }

    public double PM4_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
      return PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    }

    public double PM5_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
      return PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
    }
  }

  private static final class CS_FluorShell_Kissel_CascadeImpl extends CS_FluorShell_Cascade_Body {
  
    public double PL1_cascade_kissel(int Z, double E, double PK) {
      return PL1_full_cascade_kissel(Z, E, PK);
    }

    public double PL2_cascade_kissel(int Z, double E, double PK, double PL1) {
      return PL2_full_cascade_kissel(Z, E, PK, PL1);
    }

    public double PL3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
      return PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    }

    public double PM1_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
      return PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    }

    public double PM2_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
      return PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    }

    public double PM3_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
      return PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    }

    public double PM4_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
      return PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    }

    public double PM5_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
      return PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
    }
  }

  private static final CS_FluorShell_Cascade_Body CS_FLUORSHELL_KISSEL_RADIATIVE = new CS_FluorShell_Kissel_Radiative_CascadeImpl();
  private static final CS_FluorShell_Cascade_Body CS_FLUORSHELL_KISSEL_NONRADIATIVE = new CS_FluorShell_Kissel_Nonradiative_CascadeImpl();
  private static final CS_FluorShell_Cascade_Body CS_FLUORSHELL_KISSEL_FULL = new CS_FluorShell_Kissel_CascadeImpl();

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the radiative cascade contributions but excludes the non-radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E) {
    return CS_FLUORSHELL_KISSEL_RADIATIVE.execute(Z, shell, E);
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the non-radiative cascade contributions but excludes the radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E) {
    return CS_FLUORSHELL_KISSEL_NONRADIATIVE.execute(Z, shell, E);
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes both non-radiative and radiative cascade effect contributions!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell_Kissel_Cascade(int Z, int shell, double E) {
    return CS_FLUORSHELL_KISSEL_FULL.execute(Z, shell, E);
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes both non-radiative and radiative cascade effect contributions!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell_Kissel_Cascade(int Z, int shell, double E) {
    return CS_FluorShell_Kissel_Cascade(Z, shell, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the non-radiative cascade contributions but excludes the radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E) {
    return CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the radiative cascade contributions but excludes the non-radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E) {
    return CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method excludes the cascade effect from the calculation!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell_Kissel_no_Cascade(int Z, int shell, double E) {
    return CS_FluorShell_Kissel_no_Cascade(Z, shell, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell, and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method is an alias for #CS_FluorShell_Kissel_Cascade, meaning that the cascade effect will be taken into account.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell_Kissel(int Z, int shell, double E) {
    return CS_FluorShell_Kissel_Cascade(Z, shell, E);
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method is an alias for #CSb_FluorShell_Kissel_Cascade, meaning that the cascade effect will be taken into account.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell_Kissel(int Z, int shell, double E) {
    return CSb_FluorShell_Kissel_Cascade(Z, shell, E);
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes both non-radiative and radiative cascade effect contributions!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine_Kissel_Cascade(int Z, int line, double E) {
    return CS_FluorLine_Kissel_Cascade(Z, line, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the non-radiative cascade contributions but excludes the radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E) {
    return CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This implementation includes the radiative cascade contributions but excludes the non-radiative cascade effect!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E) {
    return CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method excludes the cascade effect from the calculation!
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine_Kissel_no_Cascade(int Z, int line, double E) {
    return CS_FluorLine_Kissel_no_Cascade(Z, line, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  public static double Jump_from_K(int Z, double E) {
    double edgeK = EdgeEnergy(Z, K_SHELL);
    double Factor;
    if (E > edgeK && edgeK > 0.0) {
      double yield;
      double JumpK = JumpFactor(Z, K_SHELL);
      yield = FluorYield(Z, K_SHELL);
      Factor = ((JumpK - 1)/JumpK) * yield;
    }
    else {
      throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
    }
    return Factor;
  }

  private static double Jump_from_L1(int Z, double E) {
    double Factor = 1.0, JumpL1, JumpK;
    double edgeK = EdgeEnergy_catch(Z, K_SHELL);
    double edgeL1 = EdgeEnergy(Z, L1_SHELL);
    double yield;

    if (E > edgeK && edgeK > 0.0) {
      JumpK = JumpFactor_catch(Z, K_SHELL);
      if (JumpK == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      Factor /= JumpK ;
    }
 
    if (E > edgeL1 && edgeL1 > 0.0) {
      JumpL1 = JumpFactor_catch(Z, L1_SHELL);
      if (JumpL1 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      yield = FluorYield_catch(Z, L1_SHELL);
      if (yield == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_FLUOR_YIELD);
      }
      Factor *= ((JumpL1 - 1) / JumpL1) * yield;
    }
    else {
      throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
    }

    return Factor;
  }

  private static double Jump_from_L2(int Z, double E) {
    double Factor = 1.0, JumpL1, JumpL2, JumpK;
    double TaoL1 = 0.0, TaoL2 = 0.0;
    double edgeK = EdgeEnergy_catch(Z, K_SHELL);
    double edgeL1 = EdgeEnergy_catch(Z, L1_SHELL);
    double edgeL2 = EdgeEnergy(Z, L2_SHELL);
    double ck_L12, yield;

    if (E > edgeK && edgeK > 0.0) {
      JumpK = JumpFactor_catch(Z, K_SHELL);
      if (JumpK == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      Factor /= JumpK ;
    }

    JumpL1 = JumpFactor_catch(Z, L1_SHELL);
    JumpL2 = JumpFactor_catch(Z, L2_SHELL);

    if (E > edgeL1 && edgeL1 > 0.0) {
      if (JumpL1 == 0.0 || JumpL2 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      TaoL1 = (JumpL1 - 1) / JumpL1 ;
      TaoL2 = (JumpL2 - 1) / (JumpL2 * JumpL1) ;
    }
    else if (E > edgeL2 && edgeL2 > 0.0) {
      if (JumpL2 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      TaoL1 = 0.0;
      TaoL2 = (JumpL2 - 1) / JumpL2;
    }
    else {
     throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
    }

    ck_L12 = CosKronTransProb_catch(Z, FL12_TRANS);
    if (TaoL1 > 0 && ck_L12 == 0.0) {
      throw new IllegalArgumentException(UNAVAILABLE_CK);
    }

    yield = FluorYield_catch(Z, L2_SHELL);
    if (yield == 0.0) {
      throw new IllegalArgumentException(UNAVAILABLE_FLUOR_YIELD);
    }

    Factor *= (TaoL2 + TaoL1 * ck_L12) * yield;

    return Factor;
  }

  private static double Jump_from_L3(int Z, double E) {
    double Factor = 1.0, JumpL1, JumpL2, JumpL3, JumpK;
    double TaoL1 = 0.0, TaoL2 = 0.0, TaoL3 = 0.0;
    double edgeK = EdgeEnergy_catch(Z, K_SHELL);
    double edgeL1 = EdgeEnergy_catch(Z, L1_SHELL);
    double edgeL2 = EdgeEnergy_catch(Z, L2_SHELL);
    double edgeL3 = EdgeEnergy(Z, L3_SHELL);
    double ck_L23, ck_L13, ck_LP13, ck_L12;
    double yield;

    if (E > edgeK && edgeK > 0.0) {
      JumpK = JumpFactor_catch(Z, K_SHELL);
      if (JumpK == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      Factor /= JumpK ;
    }
    JumpL1 = JumpFactor_catch(Z, L1_SHELL);
    JumpL2 = JumpFactor_catch(Z, L2_SHELL);
    JumpL3 = JumpFactor_catch(Z, L3_SHELL);

    if (E > edgeL1 && edgeL1 > 0.0) {
      if (JumpL1 == 0.0 || JumpL2 == 0.0 || JumpL3 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      TaoL1 = (JumpL1 - 1) / JumpL1 ;
      TaoL2 = (JumpL2 - 1) / (JumpL2 * JumpL1) ;
      TaoL3 = (JumpL3 - 1) / (JumpL3 * JumpL2 * JumpL1) ;
    }
    else if (E > edgeL2 && edgeL2 > 0.0) {
      if (JumpL2 == 0.0 || JumpL3 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      TaoL1 = 0.0;
      TaoL2 = (JumpL2 - 1) / (JumpL2) ;
      TaoL3 = (JumpL3 - 1) / (JumpL3 * JumpL2) ;
    }
    else if (E > edgeL3 && edgeL3 > 0.0) {
      TaoL1 = 0.0;
      TaoL2 = 0.0;
      if (JumpL3 == 0.0) {
        throw new IllegalArgumentException(UNAVAILABLE_JUMP_FACTOR);
      }
      TaoL3 = (JumpL3 - 1) / JumpL3 ;
    }
    else {
      throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
    }

    ck_L23 = CosKronTransProb_catch(Z, FL23_TRANS);
    ck_L13 = CosKronTransProb_catch(Z, FL13_TRANS);
    ck_LP13 = CosKronTransProb_catch(Z, FLP13_TRANS);
    ck_L12 = CosKronTransProb_catch(Z, FL12_TRANS);

    if (TaoL2 > 0.0 && ck_L23 == 0.0) {
      throw new IllegalArgumentException(UNAVAILABLE_CK);
    }

    if (TaoL1 > 0.0 && (ck_L13 + ck_LP13 == 0.0 || ck_L12 == 0.0 || ck_L23 == 0.0)) {
      throw new IllegalArgumentException(UNAVAILABLE_CK);
    }

    Factor *= TaoL3 + TaoL2 * ck_L23 + TaoL1 * (ck_L13 + ck_LP13 + ck_L12 * ck_L23);

    yield = FluorYield_catch(Z, L3_SHELL);
    if (yield == 0.0) {
      throw new IllegalArgumentException(UNAVAILABLE_FLUOR_YIELD);
    }

    Factor *= yield;
    return Factor;
  }

  private static double Jump_from_L1_catch(int Z, double E) {
     try {
       return Jump_from_L1(Z, E);
     } catch (IllegalArgumentException e) {
       return 0.0;
     }
  }

  private static double Jump_from_L2_catch(int Z, double E) {
     try {
       return Jump_from_L2(Z, E);
     } catch (IllegalArgumentException e) {
       return 0.0;
     }
  }

  private static double Jump_from_L3_catch(int Z, double E) {
     try {
       return Jump_from_L3(Z, E);
     } catch (IllegalArgumentException e) {
       return 0.0;
     }
  }

  private static double Jump_from_K_catch(int Z, double E) {
     try {
       return Jump_from_K(Z, E);
     } catch (IllegalArgumentException e) {
       return 0.0;
     }
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section for all lines that may originate from the provided shell.
   * 
   * This method used the jump factor approximation to calculate the photoionization cross section,
   * which is only supported for K- and L-lines.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorShell(int Z, int shell, double E) {
    double cs = 0.0;
    double Factor = 0.0;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    switch (shell) {
      case K_SHELL:
        Factor = Jump_from_K(Z, E);
        break;
      case L1_SHELL:
        Factor = Jump_from_L1(Z, E);
        break;
      case L2_SHELL:
        Factor = Jump_from_L2(Z, E);
        break;
      case L3_SHELL:
        Factor = Jump_from_L3(Z, E);
        break;
      default:
        throw new IllegalArgumentException(INVALID_SHELL);
    }
  
    return CS_Photo(Z, E) * Factor;
  }

  /** 
   * For a given atomic number, line and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method used the jump factor approximation to calculate the photoionization cross section,
   * which is only supported for K- and L-lines.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_FluorLine(int Z, int line, double E) {
    double Factor = 1.0;
    double rr;

    if (line >= KP5_LINE && line <= KB_LINE) {
      rr = RadRate(Z, line);
      Factor = CS_FluorShell(Z, K_SHELL, E);
      return rr * Factor;
    }
    else if ((line <= L1L2_LINE && line >= L3Q1_LINE) || line == LA_LINE) {
      rr = RadRate(Z, line);

      if (line >= L1P5_LINE && line <= L1L2_LINE) {
        Factor = CS_FluorShell(Z, L1_SHELL, E);
      }
      else if (line >= L2Q1_LINE && line <= L2L3_LINE)  {
        Factor = CS_FluorShell(Z, L2_SHELL, E);
      }
      /*
       * it's safe to use LA_LINE since it's only composed of 2 L3-lines
       */
      else if (line <= L3M1_LINE || line == LA_LINE) {
        Factor = CS_FluorShell(Z, L3_SHELL, E);
      }

      return rr * Factor;
    }
    else if (line == LB_LINE) {
      /*
       * b1->b17
       */
      double cs;
      double cs_line = Jump_from_L2_catch(Z, E) * (RadRate_catch(Z, L2M4_LINE) + RadRate_catch(Z, L2M3_LINE)) +
        Jump_from_L3_catch(Z, E) * (RadRate_catch(Z, L3N5_LINE) + RadRate_catch(Z, L3O4_LINE) + RadRate_catch(Z, L3O5_LINE) + RadRate_catch(Z, L3O45_LINE) + RadRate_catch(Z, L3N1_LINE) + RadRate_catch(Z, L3O1_LINE) + RadRate_catch(Z, L3N6_LINE) + RadRate_catch(Z, L3N7_LINE) + RadRate_catch(Z, L3N4_LINE)) +
        Jump_from_L1_catch(Z, E) * (RadRate_catch(Z, L1M3_LINE) + RadRate_catch(Z, L1M2_LINE) + RadRate_catch(Z, L1M5_LINE) + RadRate_catch(Z, L1M4_LINE));

      if (cs_line == 0.0) {
        throw new IllegalArgumentException(TOO_LOW_EXCITATION_ENERGY);
      }
      cs = CS_Photo(Z, E);
      return cs_line * cs;
    }
    else {
      throw new IllegalArgumentException(INVALID_LINE);
    }
  }

  private static double LineEnergyComposed(int Z, int line1, int line2) {
    double line_tmp1 = LineEnergy_catch(Z, line1);
    double line_tmp2 = LineEnergy_catch(Z, line2);
    double rate_tmp1 = RadRate_catch(Z, line1);
    double rate_tmp2 = RadRate_catch(Z, line2);
    double rv = line_tmp1 * rate_tmp1 + line_tmp2 * rate_tmp2;

    if (rv > 0.0) {
      return rv/(rate_tmp1 + rate_tmp2);
    }
    else if ((line_tmp1 + line_tmp2) > 0.0) {
      return (line_tmp1 + line_tmp2) / 2.0; /* in case of both radiative rates missing, use the average of both line energies. */
    }
    throw new IllegalArgumentException(INVALID_LINE);
  }

  /** 
   * For a given atomic number, line, returns the corresponding XRF line energy
   * 
   * The line energies are equal to the difference of the absorption edge energies of the two shells
   * that are involved in the transition.
   *
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @return The XRF line energy, expressed in keV.
   */
  public static double LineEnergy(int Z, int line) {
    double line_energy;
    double lE, rr;
    double tmp = 0.0, tmp1 = 0.0, tmp2 = 0.0;
    int i;
  
    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }
  
    if (line == KA_LINE || line == KB_LINE) {
      if (line == KA_LINE) {
        for (i = KL1; i <= KL3 ; i++) {
          lE = LineEnergy_arr[Z * LINENUM + i];
          rr = RadRate_arr[Z * LINENUM + i];
          tmp1 += rr;
          tmp += lE * rr;
        }
      }
      else if (line == KB_LINE) {
        for (i = KM1; i < KP5; i++) {
          lE = LineEnergy_arr[Z * LINENUM + i];
          rr = RadRate_arr[Z * LINENUM + i];
          tmp1 += rr;
          tmp += lE * rr;
        }
      }
      if (tmp1 > 0) {
        return tmp / tmp1;
      }
      else {
        throw new IllegalArgumentException(INVALID_LINE);
      }
    }
  
    if (line == LA_LINE) {
      return LineEnergyComposed(Z, L3M4_LINE, L3M5_LINE);
    }
    else if (line == LB_LINE) {
      tmp2 = 0.0;
      tmp = 0.0;

      for (LineShellPair pair : lb_pairs) {
        tmp1 = CS_FluorLine_catch(Z, pair.line, EdgeEnergy_catch(Z, pair.shell) + 0.1);
        tmp2 += tmp1;
        tmp += LineEnergy_catch(Z, pair.line) * tmp1;
      }

      if (tmp2 > 0) {
        return tmp / tmp2;
      }
      else {
        throw new IllegalArgumentException(INVALID_LINE);
      }
    }
    /*
     * special cases for composed lines
     */
    else if (line == L1N67_LINE) {
      return LineEnergyComposed(Z, L1N6_LINE, L1N7_LINE);
    }
    else if (line == L1O45_LINE) {
      return LineEnergyComposed(Z, L1O4_LINE, L1O5_LINE);
    }
    else if (line == L1P23_LINE) {
      return LineEnergyComposed(Z, L1P2_LINE, L1P3_LINE);
    }
    else if (line == L2P23_LINE) {
      return LineEnergyComposed(Z, L2P2_LINE, L2P3_LINE);
    }
    else if (line == L3O45_LINE) {
      return LineEnergyComposed(Z, L3O4_LINE, L3O5_LINE);
    }
    else if (line == L3P23_LINE) {
      return LineEnergyComposed(Z, L3O4_LINE, L3O5_LINE);
    }
    else if (line == L3P45_LINE) {
      return LineEnergyComposed(Z, L3P4_LINE, L3P5_LINE);
    }
    /*
     * KO_LINE and KP_LINE only have entries in the radrate database, not in the fluor_lines one.
     * So to get the line energies, we should map to a macro that will work as long as there is an appropriate
     * line energy in the fluor_lines database.
     */
    else if (line == KO_LINE) {
      line = KO1_LINE;
    }
    else if (line == KP_LINE) {
      line = KP1_LINE;
    }

  
    line = -line - 1;
    if (line < 0 || line >= LINENUM) {
      throw new IllegalArgumentException(UNKNOWN_LINE);
    }
  
    line_energy = LineEnergy_arr[Z * LINENUM + line];

    if (line_energy <= 0.0) {
      throw new IllegalArgumentException(INVALID_LINE);
    }
    return line_energy;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding total attenuation cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Total(int Z, double E) {
    return CS_Total(Z, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding photoionization cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Photo(int Z, double E) {
    return CS_Photo(Z, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding Rayleigh scattering cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Rayl(int Z, double E) {
    return CS_Rayl(Z, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number and energy, returns the corresponding Compton scattering cross section.
   *
   * @param Z The atomic number
   * @param E The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Compt(int Z, double E) {
    return CS_Compt(Z, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section.
   * 
   * This method used the jump factor approximation to calculate the photoionization cross section,
   * which is only support for K- and L-lines.
   * 
   * @param Z The atomic number
   * @param line A macro identifying the line, such as #KL3_LINE or #LA1_LINE.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorLine(int Z, int line, double E) {
    return CS_FluorLine(Z, line, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, shell and excitation energy, returns the corresponding XRF production cross section for all lines that may originate from the provided shell.
   * 
   * This method used the jump factor approximation to calculate the photoionization cross section,
   * which is only supported for K- and L-lines.
   * 
   * @param Z The atomic number
   * @param shell A macro identifying the shell, such as #K_SHELL or #L1_SHELL.
   * @param E The energy of the photon, expressed in keV.
   * @return The XRF production cross section, expressed in barn/atom.
   */
  public static double CSb_FluorShell(int Z, int shell, double E) {
    return CS_FluorShell(Z, shell, E) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, energy and scattering angle, returns the Rayleigh differential cross section.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSb_Rayl(int Z, double E, double theta) {
    return DCS_Rayl(Z, E, theta) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, energy and scattering angle, returns the Compton differential cross section.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSb_Compt(int Z, double E, double theta) {
    return DCS_Compt(Z, E, theta) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, energy and scattering angles, returns the Rayleigh differential cross section for a polarized beam.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSPb_Rayl(int Z, double E, double theta, double phi) {
    return DCSP_Rayl(Z, E, theta, phi) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number, energy and scattering angles, returns the Compton differential cross section for a polarized beam.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSPb_Compt(int Z, double E, double theta, double phi) {
    return DCSP_Compt(Z, E, theta, phi) * AtomicWeight_arr[Z] / AVOGNUM;
  }

  /** 
   * For a given atomic number and energy, returns the anomalous scattering factor Δf′.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @return The anomalous scattering factor Δf′
   */
  public static double Fi(int Z, double E) {
    double fi;

    if (Z < 1 || Z > ZMAX || NE_Fi_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    fi = splint(E_Fi_arr[Z], Fi_arr[Z], Fi_arr2[Z], NE_Fii_arr[Z], E);

    return fi;
  }

  /** 
   * For a given atomic number and energy, returns the anomalous scattering factor Δf″.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @return The anomalous scattering factor Δf″
   */
  public static double Fii(int Z, double E) {
    double fii;

    if (Z < 1 || Z > ZMAX || NE_Fii_arr[Z] < 0) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    fii = splint(E_Fii_arr[Z], Fii_arr[Z], Fii_arr2[Z], NE_Fii_arr[Z], E);

    return fii;
  }

  /** 
   * For a given atomic number, energy and scattering angles, returns the Rayleigh differential cross section for a polarized beam.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in cm<sup>2</sup>/g/sterad.
   */
  public static double DCSP_Rayl(int Z, double E, double theta, double phi) {
    double F, q;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    q = MomentTransf(E , theta);
    F = FF_Rayl(Z, q);
    return  AVOGNUM / AtomicWeight(Z) * F*F * DCSP_Thoms(theta, phi);
  }

  /** 
   * For a given atomic number, energy and scattering angles, returns the Compton differential cross section for a polarized beam.
   * 
   * @param Z The atomic number
   * @param E The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in cm<sup>2</sup>/g/sterad.
   */
  public static double DCSP_Compt(int Z, double E, double theta, double phi) {
    double S, q;

    if (Z < 1 || Z > ZMAX) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    q = MomentTransf(E, theta);
    S = SF_Compt(Z, q);
    return  AVOGNUM / AtomicWeight(Z) * S * DCSP_KN(E, theta, phi);
  }

  /** 
   * For a given energy and scattering angles, returns the Klein-Nishina differential cross section for a polarized beam.
   * 
   * @param E The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Klein-Nishina differential cross section, expressed in cm<sup>2</sup>/g/sterad.
   */
  public static double DCSP_KN(double E, double theta, double phi) {
    double k0_k, k_k0, k_k0_2, cos_th, sin_th, cos_phi;

    if (E <= 0.0) {
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    }

    cos_th = Math.cos(theta);
    sin_th = Math.sin(theta);
    cos_phi = Math.cos(phi);

    k0_k = 1.0 + (1.0 - cos_th) * E / MEC2 ;
    k_k0 = 1.0 / k0_k;
    k_k0_2 = k_k0 * k_k0;

    return (RE2/2.) * k_k0_2 * (k_k0 + k0_k - 2 * sin_th * sin_th
			      * cos_phi * cos_phi);
  }

  /** 
   * For the given scattering angles, returns the @see <a href="https://en.wikipedia.org/wiki/Thomson_scattering">Thomson differential cross section for a polarized beam</a>
   * 
   * @param theta The scattering angle, between indicent and observed photon or wave.
   * @param phi The azimuthal scattering angle, between indicent and observed photon or wave.
   * @return The Thomson differential cross section, expressed in barn
   */
  public static double DCSP_Thoms(double theta, double phi) {
    double sin_th, cos_phi ;

    sin_th = Math.sin(theta) ;
    cos_phi = Math.cos(phi);
    return RE2 * (1.0 - sin_th * sin_th * cos_phi * cos_phi);
  }

  /** 
   * For the given atomic number, return the corresponding chemical symbol 
   * 
   * @param Z The atomic number
   * @return The chemical symbol
   */
  public static String AtomicNumberToSymbol(int Z) {
    if (Z < 1 || Z >= MendelArray.length) {
      throw new IllegalArgumentException(Z_OUT_OF_RANGE);
    }

    return MendelArray[Z];
  }

  /** 
   * For the given chemical symbol, return the corresponding atomic number
   * 
   * @param symbol The chemical symbol
   * @return The atomic number
   */
  public static int SymbolToAtomicNumber(String symbol) {
    int i;

    if (symbol == null)
      throw new IllegalArgumentException("Invalid chemical symbol");

    for (i=1 ; i < MendelArray.length ; i++) {
      if (symbol.equals(MendelArray[i])) {
        return i;
      }
    }

    throw new IllegalArgumentException("Invalid chemical symbol");
  }

  /** 
   * Parse the given chemical compound, and return its information as a @see compoundData instance.
   * 
   * @param compoundString the chemical compound
   * @return A @see compoundData instance
   */
  public static compoundData CompoundParser(String compoundString) {
    return new compoundData(compoundString);
  }

  private static compoundDataBase parseCompoundFull(String compound) {
    try {
      return CompoundParser(compound);
    } catch (IllegalArgumentException e) {
      try {
        return GetCompoundDataNISTByName(compound);
      } catch (IllegalArgumentException e2) {
        throw new IllegalArgumentException(UNKNOWN_COMPOUND);
      }
    }
  }

  @FunctionalInterface
  private static interface CS_Body_Energy {
    public default double execute(String compound, double energy) {
      compoundDataBase cd = parseCompoundFull(compound);
      double rv = 0.0;
      for (int i = 0 ; i < cd.getNElements() ; i++)
        rv += cd.getMassFractions()[i] * CS_base(cd.getElements()[i], energy);

      return rv;
    }

    public abstract double CS_base(int Z, double energy);
  }

  @FunctionalInterface
  private static interface CS_Body_Energy_Theta {
    public default double execute(String compound, double energy, double theta) {
      compoundDataBase cd = parseCompoundFull(compound);
      double rv = 0.0;
      for (int i = 0 ; i < cd.getNElements() ; i++)
        rv += cd.getMassFractions()[i] * CS_base(cd.getElements()[i], energy, theta);

      return rv;
    }

    public abstract double CS_base(int Z, double energy, double theta);
  }

  @FunctionalInterface
  private static interface CS_Body_Energy_Theta_Phi {
    public default double execute(String compound, double energy, double theta, double phi) {
      compoundDataBase cd = parseCompoundFull(compound);
      double rv = 0.0;
      for (int i = 0 ; i < cd.getNElements() ; i++)
        rv += cd.getMassFractions()[i] * CS_base(cd.getElements()[i], energy, theta, phi);

      return rv;
    }

    public abstract double CS_base(int Z, double energy, double theta, double phi);
  }


  private static final CS_Body_Energy CS_TOTAL_CP = Xraylib::CS_Total;
  private static final CS_Body_Energy CS_TOTAL_KISSEL_CP = Xraylib::CS_Total_Kissel;
  private static final CS_Body_Energy CS_RAYL_CP = Xraylib::CS_Rayl; 
  private static final CS_Body_Energy CS_COMPT_CP = Xraylib::CS_Compt; 
  private static final CS_Body_Energy CS_PHOTO_CP = Xraylib::CS_Photo; 
  private static final CS_Body_Energy CS_PHOTO_TOTAL_CP = Xraylib::CS_Photo_Total; 
  private static final CS_Body_Energy CS_ENERGY_CP = Xraylib::CS_Energy; 

  private static final CS_Body_Energy CSB_TOTAL_CP = Xraylib::CSb_Total; 
  private static final CS_Body_Energy CSB_TOTAL_KISSEL_CP = Xraylib::CSb_Total_Kissel; 
  private static final CS_Body_Energy CSB_RAYL_CP = Xraylib::CSb_Rayl; 
  private static final CS_Body_Energy CSB_COMPT_CP = Xraylib::CSb_Compt; 
  private static final CS_Body_Energy CSB_PHOTO_CP = Xraylib::CSb_Photo; 
  private static final CS_Body_Energy CSB_PHOTO_TOTAL_CP = Xraylib::CSb_Photo_Total; 

  private static final CS_Body_Energy_Theta DCS_RAYL_CP = Xraylib::DCS_Rayl; 
  private static final CS_Body_Energy_Theta DCS_COMPT_CP = Xraylib::DCS_Compt; 
  private static final CS_Body_Energy_Theta DCSB_RAYL_CP = Xraylib::DCSb_Rayl; 
  private static final CS_Body_Energy_Theta DCSB_COMPT_CP = Xraylib::DCSb_Compt; 

  private static final CS_Body_Energy_Theta_Phi DCSP_RAYL_CP = Xraylib::DCSP_Rayl; 
  private static final CS_Body_Energy_Theta_Phi DCSP_COMPT_CP = Xraylib::DCSP_Compt; 
  private static final CS_Body_Energy_Theta_Phi DCSPB_RAYL_CP = Xraylib::DCSPb_Rayl; 
  private static final CS_Body_Energy_Theta_Phi DCSPB_COMPT_CP = Xraylib::DCSPb_Compt; 

  /** 
   * For a given compound and energy, returns the corresponding total attenuation cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Total_CP(String compound, double energy) {
    return CS_TOTAL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding total attenuation cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Total_Kissel_CP(String compound, double energy) {
    return CS_TOTAL_KISSEL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding Rayleigh scattering cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Rayl_CP(String compound, double energy) {
    return CS_RAYL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding Compton scattering cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Compt_CP(String compound, double energy) {
    return CS_COMPT_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Photo_CP(String compound, double energy) {
    return CS_PHOTO_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Photo_Total_CP(String compound, double energy) {
    return CS_PHOTO_TOTAL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding total attenuation cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Total_CP(String compound, double energy) {
    return CSB_TOTAL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding total attenuation cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Total_Kissel_CP(String compound, double energy) {
    return CSB_TOTAL_KISSEL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding Rayleigh scattering cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Rayl_CP(String compound, double energy) {
    return CSB_RAYL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding Compton scattering cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Compt_CP(String compound, double energy) {
    return CSB_COMPT_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Photo_CP(String compound, double energy) {
    return CSB_PHOTO_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding photoionization cross section.
   *
   * This method used the Kissel database to calculate the photoionization cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in barn/atom.
   */
  public static double CSb_Photo_Total_CP(String compound, double energy) {
    return CSB_PHOTO_TOTAL_CP.execute(compound, energy);
  }

  /** 
   * For a given compound and energy, returns the corresponding mass-energy absorption cross section.
   *
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The energy of the photon, expressed in keV.
   * @return The cross section, expressed in cm<sup>2</sup>/g.
   */
  public static double CS_Energy_CP(String compound, double energy) {
    return CS_ENERGY_CP.execute(compound, energy);
  }

  /** 
   * For a given compound, energy and scattering angle, returns the Rayleigh differential cross section.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in cm<sup>2</sup>/g/sterad
   */
  public static double DCS_Rayl_CP(String compound, double energy, double theta) {
    return DCS_RAYL_CP.execute(compound, energy, theta);
  }

  /** 
   * For a given compound, energy and scattering angle, returns the Compton differential cross section.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in cm<sup>2</sup>/g/sterad
   */
  public static double DCS_Compt_CP(String compound, double energy, double theta) {
    return DCS_COMPT_CP.execute(compound, energy, theta);
  }

  /** 
   * For a given compound, energy and scattering angle, returns the Rayleigh differential cross section.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSb_Rayl_CP(String compound, double energy, double theta) {
    return DCSB_RAYL_CP.execute(compound, energy, theta);
  }

  /** 
   * For a given compound, energy and scattering angle, returns the Compton differential cross section.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in barn/atom/sterad
   */
  public static double DCSb_Compt_CP(String compound, double energy, double theta) {
    return DCSB_COMPT_CP.execute(compound, energy, theta);
  }

  /** 
   * For a given compound, energy and scattering angles, returns the Rayleigh differential cross section for a polarized beam.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in cm<sup>2</sup>/g/sterad.
   */
  public static double DCSP_Rayl_CP(String compound, double energy, double theta, double phi) {
    return DCSP_RAYL_CP.execute(compound, energy, theta, phi);
  }

  /** 
   * For a given compound, energy and scattering angles, returns the Compton differential cross section for a polarized beam.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in cm<sup>2</sup>/g/sterad.
   */
  public static double DCSP_Compt_CP(String compound, double energy, double theta, double phi) {
    return DCSP_COMPT_CP.execute(compound, energy, theta, phi);
  }

  /** 
   * For a given compound, energy and scattering angles, returns the Rayleigh differential cross section for a polarized beam.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Rayleigh differential cross section, expressed in barn/atom/sterad.
   */
  public static double DCSPb_Rayl_CP(String compound, double energy, double theta, double phi) {
    return DCSPB_RAYL_CP.execute(compound, energy, theta, phi);
  }

  /** 
   * For a given compound, energy and scattering angles, returns the Compton differential cross section for a polarized beam.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param energy The photon energy, expressed in keV
   * @param theta The polar scattering angle, between indicent and observed photon.
   * @param phi The azimuthal scattering angle, between indicent and observed photon.
   * @return The Compton differential cross section, expressed in barn/atom/sterad.
   */
  public static double DCSPb_Compt_CP(String compound, double energy, double theta, double phi) {
    return DCSPB_COMPT_CP.execute(compound, energy, theta, phi);
  }

  /** 
   * For a given compound name, return the corresponding @see compoundDataNIST instance, if found in the NIST database
   * 
   * @param compoundString a valid NIST database compound name 
   * @return an instance of @see compoundDataNIST
   */
  public static compoundDataNIST GetCompoundDataNISTByName(String compoundString) {
    for (compoundDataNIST cdn : compoundDataNISTList) {
      if (cdn.name.equals(compoundString))
        return new compoundDataNIST(cdn);
    }
    throw new IllegalArgumentException(String.format("%s was not found in the NIST compound database", compoundString));
  }

  /** 
   * For a given index, return the corresponding @see compoundDataNIST instance, if found in the NIST database
   * 
   * @param compoundIndex the index at which the requested compound is stored the database
   * @return an instance of @see compoundDataNIST
   */
  public static compoundDataNIST GetCompoundDataNISTByIndex(int compoundIndex) {
    if (compoundIndex < 0 || compoundIndex >= compoundDataNISTList.length) {
      throw new IllegalArgumentException(String.format("%d is out of the range of indices covered by the NIST compound database", compoundIndex));
    }
    return new compoundDataNIST(compoundDataNISTList[compoundIndex]);
  }

  /** 
   * @return a list of all compound names present in the NIST database
   */
  public static String[] GetCompoundDataNISTList() {
    return Arrays.stream(compoundDataNISTList).map(compound -> compound.name).toArray(String[]::new);
  }

  /** 
   * For a given radionuclide name, return the corresponding @see radioNuclideData instance, if found in the database
   * 
   * @param radioNuclideString a valid radionuclide name
   * @return an instance of @see radioNuclideData
   */
  public static radioNuclideData GetRadioNuclideDataByName(String radioNuclideString) {
    for (radioNuclideData rnd : nuclideDataList) {
      if (rnd.name.equals(radioNuclideString))
        return new radioNuclideData(rnd);
    }
    throw new IllegalArgumentException(String.format("%s was not found in the radionuclide database", radioNuclideString));
  }

  /** 
   * For a given index, return the corresponding @see radioNuclideData instance, if found in the database
   * 
   * @param radioNuclideIndex the index at which the requested radionuclide is stored the database
   * @return an instance of @see radioNuclideData
   */
  public static radioNuclideData GetRadioNuclideDataByIndex(int radioNuclideIndex) {
    if (radioNuclideIndex < 0 || radioNuclideIndex >= nuclideDataList.length) {
      throw new IllegalArgumentException(String.format("%d is out of the range of indices covered by the radionuclide database", radioNuclideIndex));
    }
    return new radioNuclideData(nuclideDataList[radioNuclideIndex]);
  }

  /** 
   * @return a list of all radionuclide names present in the database
   */
  public static String[] GetRadioNuclideDataList() {
    return Arrays.stream(nuclideDataList).map(nuclide -> nuclide.name).toArray(String[]::new);
  }

  private static final double KD = 4.15179082788e-4;

  /** 
   * For a given compound, energy and density, returns the real part of the refractive index.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param E The photon energy, expressed in keV
   * @param density The density, expressed in g/cm<sup>3</sup>. If the compound is found in the NIST database, this argument will only be used if it is greater than zero, otherwise the density from the database will used instead.
   * @return The real part of the refractive index
   */
  public static double Refractive_Index_Re(String compound, double E, double density) {
    compoundDataBase cd = parseCompoundFull(compound);
    double delta = 0.0;

    if (E <= 0.0)
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    else if (density <= 0.0) {
      if (compoundDataNIST.class.isInstance(cd)) {
        density = compoundDataNIST.class.cast(cd).density;
      } else {
        throw new IllegalArgumentException(NEGATIVE_DENSITY);
      }
    }

    for (int i = 0 ; i < cd.getNElements() ; i++)
      delta += cd.getMassFractions()[i] * KD * (cd.getElements()[i] + Fi(cd.getElements()[i], E)) / AtomicWeight(cd.getElements()[i]) / E / E;

    return 1.0 - delta * density;
  }

  /** 
   * For a given compound, energy and density, returns the imaginary part of the refractive index.
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param E The photon energy, expressed in keV
   * @param density The density, expressed in g/cm<sup>3</sup>. If the compound is found in the NIST database, this argument will only be used if it is greater than zero, otherwise the density from the database will used instead.
   * @return The imaginary part of the refractive index
   */
  public static double Refractive_Index_Im(String compound, double E, double density) {
    compoundDataBase cd = parseCompoundFull(compound);
    double rv = 0.0;

    if (E <= 0.0)
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    else if (density <= 0.0) {
      if (compoundDataNIST.class.isInstance(cd)) {
        density = compoundDataNIST.class.cast(cd).density;
      } else {
        throw new IllegalArgumentException(NEGATIVE_DENSITY);
      }
    }

    for (int i = 0 ; i < cd.getNElements() ; i++)
      rv += CS_Total(cd.getElements()[i], E) * cd.getMassFractions()[i];

    return rv * density * 9.8663479e-9 / E;
  }

  /** 
   * For a given compound, energy and density, returns the refractive index as a complex number .
   * 
   * @param compound Either a valid chemical formula, or a compound from the NIST database (see #GetCompoundDataNISTList for a list)
   * @param E The photon energy, expressed in keV
   * @param density The density, expressed in g/cm<sup>3</sup>. If the compound is found in the NIST database, this argument will only be used if it is greater than zero, otherwise the density from the database will used instead.
   * @return The refractive index as a complex number
   */
  public static Complex Refractive_Index(String compound, double E, double density) {
    double re = 0.0;
    double im = 0.0;
    compoundDataBase cd = parseCompoundFull(compound);

    if (E <= 0.0)
      throw new IllegalArgumentException(NEGATIVE_ENERGY);
    else if (density <= 0.0) {
      if (compoundDataNIST.class.isInstance(cd)) {
        density = compoundDataNIST.class.cast(cd).density;
      } else {
        throw new IllegalArgumentException(NEGATIVE_DENSITY);
      }
    }

    for (int i = 0 ; i < cd.getNElements() ; i++) {
      re += cd.getMassFractions()[i] * KD * (cd.getElements()[i] + Fi(cd.getElements()[i], E)) / AtomicWeight(cd.getElements()[i]) / E / E;
      im += CS_Total(cd.getElements()[i], E) * cd.getMassFractions()[i];
    }

    re = 1.0 - re * density;
    im = im * density * 9.8663479e-9 / E;

    return Complex.ofCartesian(re, im);
  }

  /** 
   * @return a list of all currently available crystals
   */
  public static String[] Crystal_GetCrystalsList() {
    return Arrays.stream(crystalDataList).map(crystal -> crystal.name).toArray(String[]::new);
  }

  /** 
   * For a given crystal name, returns the corresponding @see Crystal_Struct instance
   * 
   * @param material The name of the crystal
   * @return an instance of @see Crystal_Struct
   */
  public static Crystal_Struct Crystal_GetCrystal(String material) {
    for (Crystal_Struct cs : crystalDataList) {
      if (cs.name.equals(material))
        return new Crystal_Struct(cs);
    }
    throw new IllegalArgumentException(String.format("Crystal %s is not present in the array", material));
  }

  /** Calculates the Bragg angle, given an energy and set of Miller indices
   *  
   * @param cs an instance of @see Crystal_Struct
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @return the Bragg angle
   */
  public static double Bragg_angle(Crystal_Struct cs, double energy, int i_miller, int j_miller, int k_miller) {
    return cs.Bragg_angle(energy, i_miller, j_miller, k_miller);
  }

  /** Calculates the crystal structure factor
   *  
   * @param cs an instance of @see Crystal_Struct
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param debye_factor The Debye factor
   * @param rel_angle expressed in radians
   * @return the crystal structure factor, as a complex number
   */
  public static Complex Crystal_F_H_StructureFactor (Crystal_Struct cs, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle) {
    return cs.Crystal_F_H_StructureFactor(energy, i_miller, j_miller, k_miller, debye_factor, rel_angle);
  }

  /** Calculates the partial crystal structure factor
   *  
   * @param cs an instance of @see Crystal_Struct
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param debye_factor The Debye factor
   * @param rel_angle expressed in radians
   * @param f0_flag 
   * @param f_prime_flag
   * @param f_prime2_flag
   * @return the crystal structure factor, as a complex number
   */
  public static Complex Crystal_F_H_StructureFactor_Partial(Crystal_Struct cs, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
                      int f0_flag, int f_prime_flag, int f_prime2_flag) {
    return cs.Crystal_F_H_StructureFactor_Partial(energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag);
  }

  /** Calculate the unit cell volume 
   *  
   * @param cs an instance of @see Crystal_Struct
   * @return The unit cell volume
   */
  public static double Crystal_UnitCellVolume(Crystal_Struct cs) {
    return cs.Crystal_UnitCellVolume();
  }

  /** Calculates the d-spacing for the crystal and Miller indices.
   *  
   * @param cs an instance of @see Crystal_Struct
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @return The crystal D-spacing
   */
  public static double Crystal_dSpacing(Crystal_Struct cs, int i_miller, int j_miller, int k_miller) {
    return cs.Crystal_dSpacing(i_miller, j_miller, k_miller);
  }

  /** Calculates the Q scattering amplitude, given an energy, Miller indices and relative angle
   *  
   * @param cs an instance of @see Crystal_Struct
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param rel_angle expressed in radians
   * @return The Q scattering amplitude
   */
  public static double Q_scattering_amplitude(Crystal_Struct cs, double energy, int i_miller, int j_miller, int k_miller, double rel_angle) {
    return cs.Q_scattering_amplitude(energy, i_miller, j_miller, k_miller, rel_angle);
  }

  /** 
   * For a given atomic number, energy, momentum transfer and Debye factor, returns
   * an array with the atomic factors f<sub>0</sub>, Δf′ and Δf″. 
   * 
   * @param Z The atomic number
   * @param energy The energy of the photon, expressed in keV.
   * @param q The momentum transfer
   * @param debye_factor The Debye factor
   * @return an array with the three atomic factors
   */
  public static double[] Atomic_Factors(int Z, double energy, double q, double debye_factor) {
    if (debye_factor <= 0.0)
      throw new IllegalArgumentException(NEGATIVE_DEBYE_FACTOR);

    double f0 = FF_Rayl(Z, q) * debye_factor;
    double f_prime = Fi(Z, energy) * debye_factor;
    double f_prime2 = -Fii(Z, energy) * debye_factor;
    return new double[]{f0, f_prime, f_prime2};
  }

  private static double splint(double[] xa, double[] ya, double[] y2a, int n, double x) {
    int klo, khi, k;
    double h, b, a;

    if (x - xa[n-1] > 1E-7) {
      throw new IllegalArgumentException(SPLINT_X_TOO_HIGH);
    }

    if (x < xa[0]) {
      throw new IllegalArgumentException(SPLINT_X_TOO_LOW);
    }

    klo = 0;
    khi = n-1;
    while (khi-klo > 1) {
      k = (khi + klo) >> 1;
      if (xa[k] > x) {
        khi = k;
      }
      else {
        klo = k;
      }
    }

    h = xa[khi] - xa[klo];
    if (h == 0.0) {
      return (ya[klo] + ya[khi])/2.0;
    }
    a = (xa[khi] - x) / h;
    b = (x - xa[klo]) / h;
    return a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
       + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
  }

  private static double[] AtomicWeight_arr;
  private static double[] ElementDensity_arr;
  private static double[] EdgeEnergy_arr;
  private static double[] AtomicLevelWidth_arr;
  private static double[] LineEnergy_arr;
  private static double[] FluorYield_arr;
  private static double[] JumpFactor_arr;
  private static double[] CosKron_arr;
  private static double[] RadRate_arr;
  private static double[] xrf_cross_sections_constants_full;
  private static double[] xrf_cross_sections_constants_auger_only;

  private static int[] NE_Photo_arr;
  private static double[][] E_Photo_arr;
  private static double[][] CS_Photo_arr;
  private static double[][] CS_Photo_arr2;

  private static int[] NE_Rayl_arr;
  private static double[][] E_Rayl_arr;
  private static double[][] CS_Rayl_arr;
  private static double[][] CS_Rayl_arr2;

  private static int[] NE_Compt_arr;
  private static double[][] E_Compt_arr;
  private static double[][] CS_Compt_arr;
  private static double[][] CS_Compt_arr2;

  private static int[] NE_Energy_arr;
  private static double[][] E_Energy_arr;
  private static double[][] CS_Energy_arr;
  private static double[][] CS_Energy_arr2;

  private static int[] Nq_Rayl_arr;
  private static double[][] q_Rayl_arr;
  private static double[][] FF_Rayl_arr;
  private static double[][] FF_Rayl_arr2;

  private static int[] Nq_Compt_arr;
  private static double[][] q_Compt_arr;
  private static double[][] SF_Compt_arr;
  private static double[][] SF_Compt_arr2;

  private static int[] NE_Fi_arr;
  private static double[][] E_Fi_arr;
  private static double[][] Fi_arr;
  private static double[][] Fi_arr2;

  private static int[] NE_Fii_arr;
  private static double[][] E_Fii_arr;
  private static double[][] Fii_arr;
  private static double[][] Fii_arr2;

  private static int[] NE_Photo_Total_Kissel_arr;

  private static double[] Electron_Config_Kissel_arr;
  private static double[] EdgeEnergy_Kissel_arr;

  private static int[][] NE_Photo_Partial_Kissel_arr;
  private static double[][][] E_Photo_Partial_Kissel_arr;
  private static double[][][] Photo_Partial_Kissel_arr;
  private static double[][][] Photo_Partial_Kissel_arr2;

  private static int[] NShells_ComptonProfiles_arr;
  private static int[] Npz_ComptonProfiles_arr;
  private static double[][] UOCCUP_ComptonProfiles_arr;
  private static double[][] pz_ComptonProfiles_arr;
  private static double[][] Total_ComptonProfiles_arr;
  private static double[][] Total_ComptonProfiles_arr2;
  private static double[][][] Partial_ComptonProfiles_arr;
  private static double[][][] Partial_ComptonProfiles_arr2;

  private static double[] Auger_Yields_arr;
  private static double[] Auger_Rates_arr;

  private static compoundDataNIST[] compoundDataNISTList;
  private static radioNuclideData[] nuclideDataList;
  private static Crystal_Struct[] crystalDataList;

  public static int ZMAX;
  public static int SHELLNUM;
  public static int SHELLNUM_K;
  public static int SHELLNUM_A;
  public static int TRANSNUM;
  public static int LINENUM;
  public static int AUGERNUM;
  public static double RE2;
  public static double MEC2;
  public static double AVOGNUM;
  public static double KEV2ANGST;
  public static double R_E;

  public static final int K_SHELL = 0;
  public static final int L1_SHELL = 1;
  public static final int L2_SHELL = 2;
  public static final int L3_SHELL = 3;
  public static final int M1_SHELL = 4;
  public static final int M2_SHELL = 5;
  public static final int M3_SHELL = 6;
  public static final int M4_SHELL = 7;
  public static final int M5_SHELL = 8;
  public static final int N1_SHELL = 9;
  public static final int N2_SHELL = 10;
  public static final int N3_SHELL = 11;
  public static final int N4_SHELL = 12;
  public static final int N5_SHELL = 13;
  public static final int N6_SHELL = 14;
  public static final int N7_SHELL = 15;
  public static final int O1_SHELL = 16;
  public static final int O2_SHELL = 17;
  public static final int O3_SHELL = 18;
  public static final int O4_SHELL = 19;
  public static final int O5_SHELL = 20;
  public static final int O6_SHELL = 21;
  public static final int O7_SHELL = 22;
  public static final int P1_SHELL = 23;
  public static final int P2_SHELL = 24;
  public static final int P3_SHELL = 25;
  public static final int P4_SHELL = 26;
  public static final int P5_SHELL = 27;
  public static final int Q1_SHELL = 28;
  public static final int Q2_SHELL = 29;
  public static final int Q3_SHELL = 30;

  public static final int KL1_LINE = -1;
  public static final int KL2_LINE = -2;
  public static final int KL3_LINE = -3;
  public static final int KM1_LINE = -4;
  public static final int KM2_LINE = -5;
  public static final int KM3_LINE = -6;
  public static final int KM4_LINE = -7;
  public static final int KM5_LINE = -8;
  public static final int KN1_LINE = -9;
  public static final int KN2_LINE = -10;
  public static final int KN3_LINE = -11;
  public static final int KN4_LINE = -12;
  public static final int KN5_LINE = -13;
  public static final int KN6_LINE = -14;
  public static final int KN7_LINE = -15;
  public static final int KO_LINE = -16;
  public static final int KO1_LINE = -17;
  public static final int KO2_LINE = -18;
  public static final int KO3_LINE = -19;
  public static final int KO4_LINE = -20;
  public static final int KO5_LINE = -21;
  public static final int KO6_LINE = -22;
  public static final int KO7_LINE = -23;
  public static final int KP_LINE = -24;
  public static final int KP1_LINE = -25;
  public static final int KP2_LINE = -26;
  public static final int KP3_LINE = -27;
  public static final int KP4_LINE = -28;
  public static final int KP5_LINE = -29;
  public static final int L1L2_LINE = -30;
  public static final int L1L3_LINE = -31;
  public static final int L1M1_LINE = -32;
  public static final int L1M2_LINE = -33;
  public static final int L1M3_LINE = -34;
  public static final int L1M4_LINE = -35;
  public static final int L1M5_LINE = -36;
  public static final int L1N1_LINE = -37;
  public static final int L1N2_LINE = -38;
  public static final int L1N3_LINE = -39;
  public static final int L1N4_LINE = -40;
  public static final int L1N5_LINE = -41;
  public static final int L1N6_LINE = -42;
  public static final int L1N67_LINE = -43;
  public static final int L1N7_LINE = -44;
  public static final int L1O1_LINE = -45;
  public static final int L1O2_LINE = -46;
  public static final int L1O3_LINE = -47;
  public static final int L1O4_LINE = -48;
  public static final int L1O45_LINE = -49;
  public static final int L1O5_LINE = -50;
  public static final int L1O6_LINE = -51;
  public static final int L1O7_LINE = -52;
  public static final int L1P1_LINE = -53;
  public static final int L1P2_LINE = -54;
  public static final int L1P23_LINE = -55;
  public static final int L1P3_LINE = -56;
  public static final int L1P4_LINE = -57;
  public static final int L1P5_LINE = -58;
  public static final int L2L3_LINE = -59;
  public static final int L2M1_LINE = -60;
  public static final int L2M2_LINE = -61;
  public static final int L2M3_LINE = -62;
  public static final int L2M4_LINE = -63;
  public static final int L2M5_LINE = -64;
  public static final int L2N1_LINE = -65;
  public static final int L2N2_LINE = -66;
  public static final int L2N3_LINE = -67;
  public static final int L2N4_LINE = -68;
  public static final int L2N5_LINE = -69;
  public static final int L2N6_LINE = -70;
  public static final int L2N7_LINE = -71;
  public static final int L2O1_LINE = -72;
  public static final int L2O2_LINE = -73;
  public static final int L2O3_LINE = -74;
  public static final int L2O4_LINE = -75;
  public static final int L2O5_LINE = -76;
  public static final int L2O6_LINE = -77;
  public static final int L2O7_LINE = -78;
  public static final int L2P1_LINE = -79;
  public static final int L2P2_LINE = -80;
  public static final int L2P23_LINE = -81;
  public static final int L2P3_LINE = -82;
  public static final int L2P4_LINE = -83;
  public static final int L2P5_LINE = -84;
  public static final int L2Q1_LINE = -85;
  public static final int L3M1_LINE = -86;
  public static final int L3M2_LINE = -87;
  public static final int L3M3_LINE = -88;
  public static final int L3M4_LINE = -89;
  public static final int L3M5_LINE = -90;
  public static final int L3N1_LINE = -91;
  public static final int L3N2_LINE = -92;
  public static final int L3N3_LINE = -93;
  public static final int L3N4_LINE = -94;
  public static final int L3N5_LINE = -95;
  public static final int L3N6_LINE = -96;
  public static final int L3N7_LINE = -97;
  public static final int L3O1_LINE = -98;
  public static final int L3O2_LINE = -99;
  public static final int L3O3_LINE = -100;
  public static final int L3O4_LINE = -101;
  public static final int L3O45_LINE = -102;
  public static final int L3O5_LINE = -103;
  public static final int L3O6_LINE = -104;
  public static final int L3O7_LINE = -105;
  public static final int L3P1_LINE = -106;
  public static final int L3P2_LINE = -107;
  public static final int L3P23_LINE = -108;
  public static final int L3P3_LINE = -109;
  public static final int L3P4_LINE = -110;
  public static final int L3P45_LINE = -111;
  public static final int L3P5_LINE = -112;
  public static final int L3Q1_LINE = -113;
  public static final int M1M2_LINE = -114;
  public static final int M1M3_LINE = -115;
  public static final int M1M4_LINE = -116;
  public static final int M1M5_LINE = -117;
  public static final int M1N1_LINE = -118;
  public static final int M1N2_LINE = -119;
  public static final int M1N3_LINE = -120;
  public static final int M1N4_LINE = -121;
  public static final int M1N5_LINE = -122;
  public static final int M1N6_LINE = -123;
  public static final int M1N7_LINE = -124;
  public static final int M1O1_LINE = -125;
  public static final int M1O2_LINE = -126;
  public static final int M1O3_LINE = -127;
  public static final int M1O4_LINE = -128;
  public static final int M1O5_LINE = -129;
  public static final int M1O6_LINE = -130;
  public static final int M1O7_LINE = -131;
  public static final int M1P1_LINE = -132;
  public static final int M1P2_LINE = -133;
  public static final int M1P3_LINE = -134;
  public static final int M1P4_LINE = -135;
  public static final int M1P5_LINE = -136;
  public static final int M2M3_LINE = -137;
  public static final int M2M4_LINE = -138;
  public static final int M2M5_LINE = -139;
  public static final int M2N1_LINE = -140;
  public static final int M2N2_LINE = -141;
  public static final int M2N3_LINE = -142;
  public static final int M2N4_LINE = -143;
  public static final int M2N5_LINE = -144;
  public static final int M2N6_LINE = -145;
  public static final int M2N7_LINE = -146;
  public static final int M2O1_LINE = -147;
  public static final int M2O2_LINE = -148;
  public static final int M2O3_LINE = -149;
  public static final int M2O4_LINE = -150;
  public static final int M2O5_LINE = -151;
  public static final int M2O6_LINE = -152;
  public static final int M2O7_LINE = -153;
  public static final int M2P1_LINE = -154;
  public static final int M2P2_LINE = -155;
  public static final int M2P3_LINE = -156;
  public static final int M2P4_LINE = -157;
  public static final int M2P5_LINE = -158;
  public static final int M3M4_LINE = -159;
  public static final int M3M5_LINE = -160;
  public static final int M3N1_LINE = -161;
  public static final int M3N2_LINE = -162;
  public static final int M3N3_LINE = -163;
  public static final int M3N4_LINE = -164;
  public static final int M3N5_LINE = -165;
  public static final int M3N6_LINE = -166;
  public static final int M3N7_LINE = -167;
  public static final int M3O1_LINE = -168;
  public static final int M3O2_LINE = -169;
  public static final int M3O3_LINE = -170;
  public static final int M3O4_LINE = -171;
  public static final int M3O5_LINE = -172;
  public static final int M3O6_LINE = -173;
  public static final int M3O7_LINE = -174;
  public static final int M3P1_LINE = -175;
  public static final int M3P2_LINE = -176;
  public static final int M3P3_LINE = -177;
  public static final int M3P4_LINE = -178;
  public static final int M3P5_LINE = -179;
  public static final int M3Q1_LINE = -180;
  public static final int M4M5_LINE = -181;
  public static final int M4N1_LINE = -182;
  public static final int M4N2_LINE = -183;
  public static final int M4N3_LINE = -184;
  public static final int M4N4_LINE = -185;
  public static final int M4N5_LINE = -186;
  public static final int M4N6_LINE = -187;
  public static final int M4N7_LINE = -188;
  public static final int M4O1_LINE = -189;
  public static final int M4O2_LINE = -190;
  public static final int M4O3_LINE = -191;
  public static final int M4O4_LINE = -192;
  public static final int M4O5_LINE = -193;
  public static final int M4O6_LINE = -194;
  public static final int M4O7_LINE = -195;
  public static final int M4P1_LINE = -196;
  public static final int M4P2_LINE = -197;
  public static final int M4P3_LINE = -198;
  public static final int M4P4_LINE = -199;
  public static final int M4P5_LINE = -200;
  public static final int M5N1_LINE = -201;
  public static final int M5N2_LINE = -202;
  public static final int M5N3_LINE = -203;
  public static final int M5N4_LINE = -204;
  public static final int M5N5_LINE = -205;
  public static final int M5N6_LINE = -206;
  public static final int M5N7_LINE = -207;
  public static final int M5O1_LINE = -208;
  public static final int M5O2_LINE = -209;
  public static final int M5O3_LINE = -210;
  public static final int M5O4_LINE = -211;
  public static final int M5O5_LINE = -212;
  public static final int M5O6_LINE = -213;
  public static final int M5O7_LINE = -214;
  public static final int M5P1_LINE = -215;
  public static final int M5P2_LINE = -216;
  public static final int M5P3_LINE = -217;
  public static final int M5P4_LINE = -218;
  public static final int M5P5_LINE = -219;
  public static final int N1N2_LINE = -220;
  public static final int N1N3_LINE = -221;
  public static final int N1N4_LINE = -222;
  public static final int N1N5_LINE = -223;
  public static final int N1N6_LINE = -224;
  public static final int N1N7_LINE = -225;
  public static final int N1O1_LINE = -226;
  public static final int N1O2_LINE = -227;
  public static final int N1O3_LINE = -228;
  public static final int N1O4_LINE = -229;
  public static final int N1O5_LINE = -230;
  public static final int N1O6_LINE = -231;
  public static final int N1O7_LINE = -232;
  public static final int N1P1_LINE = -233;
  public static final int N1P2_LINE = -234;
  public static final int N1P3_LINE = -235;
  public static final int N1P4_LINE = -236;
  public static final int N1P5_LINE = -237;
  public static final int N2N3_LINE = -238;
  public static final int N2N4_LINE = -239;
  public static final int N2N5_LINE = -240;
  public static final int N2N6_LINE = -241;
  public static final int N2N7_LINE = -242;
  public static final int N2O1_LINE = -243;
  public static final int N2O2_LINE = -244;
  public static final int N2O3_LINE = -245;
  public static final int N2O4_LINE = -246;
  public static final int N2O5_LINE = -247;
  public static final int N2O6_LINE = -248;
  public static final int N2O7_LINE = -249;
  public static final int N2P1_LINE = -250;
  public static final int N2P2_LINE = -251;
  public static final int N2P3_LINE = -252;
  public static final int N2P4_LINE = -253;
  public static final int N2P5_LINE = -254;
  public static final int N3N4_LINE = -255;
  public static final int N3N5_LINE = -256;
  public static final int N3N6_LINE = -257;
  public static final int N3N7_LINE = -258;
  public static final int N3O1_LINE = -259;
  public static final int N3O2_LINE = -260;
  public static final int N3O3_LINE = -261;
  public static final int N3O4_LINE = -262;
  public static final int N3O5_LINE = -263;
  public static final int N3O6_LINE = -264;
  public static final int N3O7_LINE = -265;
  public static final int N3P1_LINE = -266;
  public static final int N3P2_LINE = -267;
  public static final int N3P3_LINE = -268;
  public static final int N3P4_LINE = -269;
  public static final int N3P5_LINE = -270;
  public static final int N4N5_LINE = -271;
  public static final int N4N6_LINE = -272;
  public static final int N4N7_LINE = -273;
  public static final int N4O1_LINE = -274;
  public static final int N4O2_LINE = -275;
  public static final int N4O3_LINE = -276;
  public static final int N4O4_LINE = -277;
  public static final int N4O5_LINE = -278;
  public static final int N4O6_LINE = -279;
  public static final int N4O7_LINE = -280;
  public static final int N4P1_LINE = -281;
  public static final int N4P2_LINE = -282;
  public static final int N4P3_LINE = -283;
  public static final int N4P4_LINE = -284;
  public static final int N4P5_LINE = -285;
  public static final int N5N6_LINE = -286;
  public static final int N5N7_LINE = -287;
  public static final int N5O1_LINE = -288;
  public static final int N5O2_LINE = -289;
  public static final int N5O3_LINE = -290;
  public static final int N5O4_LINE = -291;
  public static final int N5O5_LINE = -292;
  public static final int N5O6_LINE = -293;
  public static final int N5O7_LINE = -294;
  public static final int N5P1_LINE = -295;
  public static final int N5P2_LINE = -296;
  public static final int N5P3_LINE = -297;
  public static final int N5P4_LINE = -298;
  public static final int N5P5_LINE = -299;
  public static final int N6N7_LINE = -300;
  public static final int N6O1_LINE = -301;
  public static final int N6O2_LINE = -302;
  public static final int N6O3_LINE = -303;
  public static final int N6O4_LINE = -304;
  public static final int N6O5_LINE = -305;
  public static final int N6O6_LINE = -306;
  public static final int N6O7_LINE = -307;
  public static final int N6P1_LINE = -308;
  public static final int N6P2_LINE = -309;
  public static final int N6P3_LINE = -310;
  public static final int N6P4_LINE = -311;
  public static final int N6P5_LINE = -312;
  public static final int N7O1_LINE = -313;
  public static final int N7O2_LINE = -314;
  public static final int N7O3_LINE = -315;
  public static final int N7O4_LINE = -316;
  public static final int N7O5_LINE = -317;
  public static final int N7O6_LINE = -318;
  public static final int N7O7_LINE = -319;
  public static final int N7P1_LINE = -320;
  public static final int N7P2_LINE = -321;
  public static final int N7P3_LINE = -322;
  public static final int N7P4_LINE = -323;
  public static final int N7P5_LINE = -324;
  public static final int O1O2_LINE = -325;
  public static final int O1O3_LINE = -326;
  public static final int O1O4_LINE = -327;
  public static final int O1O5_LINE = -328;
  public static final int O1O6_LINE = -329;
  public static final int O1O7_LINE = -330;
  public static final int O1P1_LINE = -331;
  public static final int O1P2_LINE = -332;
  public static final int O1P3_LINE = -333;
  public static final int O1P4_LINE = -334;
  public static final int O1P5_LINE = -335;
  public static final int O2O3_LINE = -336;
  public static final int O2O4_LINE = -337;
  public static final int O2O5_LINE = -338;
  public static final int O2O6_LINE = -339;
  public static final int O2O7_LINE = -340;
  public static final int O2P1_LINE = -341;
  public static final int O2P2_LINE = -342;
  public static final int O2P3_LINE = -343;
  public static final int O2P4_LINE = -344;
  public static final int O2P5_LINE = -345;
  public static final int O3O4_LINE = -346;
  public static final int O3O5_LINE = -347;
  public static final int O3O6_LINE = -348;
  public static final int O3O7_LINE = -349;
  public static final int O3P1_LINE = -350;
  public static final int O3P2_LINE = -351;
  public static final int O3P3_LINE = -352;
  public static final int O3P4_LINE = -353;
  public static final int O3P5_LINE = -354;
  public static final int O4O5_LINE = -355;
  public static final int O4O6_LINE = -356;
  public static final int O4O7_LINE = -357;
  public static final int O4P1_LINE = -358;
  public static final int O4P2_LINE = -359;
  public static final int O4P3_LINE = -360;
  public static final int O4P4_LINE = -361;
  public static final int O4P5_LINE = -362;
  public static final int O5O6_LINE = -363;
  public static final int O5O7_LINE = -364;
  public static final int O5P1_LINE = -365;
  public static final int O5P2_LINE = -366;
  public static final int O5P3_LINE = -367;
  public static final int O5P4_LINE = -368;
  public static final int O5P5_LINE = -369;
  public static final int O6O7_LINE = -370;
  public static final int O6P4_LINE = -371;
  public static final int O6P5_LINE = -372;
  public static final int O7P4_LINE = -373;
  public static final int O7P5_LINE = -374;
  public static final int P1P2_LINE = -375;
  public static final int P1P3_LINE = -376;
  public static final int P1P4_LINE = -377;
  public static final int P1P5_LINE = -378;
  public static final int P2P3_LINE = -379;
  public static final int P2P4_LINE = -380;
  public static final int P2P5_LINE = -381;
  public static final int P3P4_LINE = -382;
  public static final int P3P5_LINE = -383;

  public static final int F1_TRANS = 0;
  public static final int F12_TRANS = 1;
  public static final int F13_TRANS = 2;
  public static final int FP13_TRANS = 3;
  public static final int F23_TRANS = 4;

  public static final int FL12_TRANS = 1;
  public static final int FL13_TRANS = 2;
  public static final int FLP13_TRANS = 3;
  public static final int FL23_TRANS = 4;
  public static final int FM12_TRANS = 5;
  public static final int FM13_TRANS = 6;
  public static final int FM14_TRANS = 7;
  public static final int FM15_TRANS = 8;
  public static final int FM23_TRANS = 9;
  public static final int FM24_TRANS = 10;
  public static final int FM25_TRANS = 11;
  public static final int FM34_TRANS = 12;
  public static final int FM35_TRANS = 13;
  public static final int FM45_TRANS = 14;

/*
 * Siegbahn notation
 * according to Table VIII.2 from Nomenclature system for X-ray spectroscopy
 * Linegroups -> usage is discouraged
 *
 */
  public static final int KA_LINE = 0;
  public static final int KB_LINE = 1;
  public static final int LA_LINE = 2;
  public static final int LB_LINE = 3;

/* single lines */
  public static final int KA1_LINE = KL3_LINE;
  public static final int KA2_LINE = KL2_LINE;
  public static final int KA3_LINE = KL1_LINE;
  public static final int KB1_LINE = KM3_LINE;
  public static final int KB2_LINE = KN3_LINE;
  public static final int KB3_LINE = KM2_LINE;
  public static final int KB4_LINE = KN5_LINE;
  public static final int KB5_LINE = KM5_LINE;

  public static final int LA1_LINE = L3M5_LINE;
  public static final int LA2_LINE = L3M4_LINE;
  public static final int LB1_LINE = L2M4_LINE;
  public static final int LB2_LINE = L3N5_LINE;
  public static final int LB3_LINE = L1M3_LINE;
  public static final int LB4_LINE = L1M2_LINE;
  public static final int LB5_LINE = L3O45_LINE;
  public static final int LB6_LINE = L3N1_LINE;
  public static final int LB7_LINE = L3O1_LINE;
  public static final int LB9_LINE = L1M5_LINE;
  public static final int LB10_LINE = L1M4_LINE;
  public static final int LB15_LINE = L3N4_LINE;
  public static final int LB17_LINE = L2M3_LINE;
  public static final int LG1_LINE = L2N4_LINE;
  public static final int LG2_LINE = L1N2_LINE;
  public static final int LG3_LINE = L1N3_LINE;
  public static final int LG4_LINE = L1O3_LINE;
  public static final int LG5_LINE = L2N1_LINE;
  public static final int LG6_LINE = L2O4_LINE;
  public static final int LG8_LINE = L2O1_LINE;
  public static final int LE_LINE = L2M1_LINE;
  public static final int LH_LINE = L2M1_LINE;
  public static final int LL_LINE = L3M1_LINE;
  public static final int LS_LINE = L3M3_LINE;
  public static final int LT_LINE = L3M2_LINE;
  public static final int LU_LINE = L3N6_LINE;
  public static final int LV_LINE = L2N6_LINE;

  public static final int MA1_LINE = M5N7_LINE;
  public static final int MA2_LINE = M5N6_LINE;
  public static final int MB_LINE = M4N6_LINE;
  public static final int MG_LINE = M3N5_LINE;

  public static final int  K_L1L1_AUGER =    0;
  public static final int  K_L1L2_AUGER =    1;
  public static final int  K_L1L3_AUGER =    2;
  public static final int  K_L1M1_AUGER =    3;
  public static final int  K_L1M2_AUGER =    4;
  public static final int  K_L1M3_AUGER =    5;
  public static final int  K_L1M4_AUGER =    6;
  public static final int  K_L1M5_AUGER =    7;
  public static final int  K_L1N1_AUGER =    8;
  public static final int  K_L1N2_AUGER =    9;
  public static final int  K_L1N3_AUGER =   10;
  public static final int  K_L1N4_AUGER =   11;
  public static final int  K_L1N5_AUGER =   12;
  public static final int  K_L1N6_AUGER =   13;
  public static final int  K_L1N7_AUGER =   14;
  public static final int  K_L1O1_AUGER =   15;
  public static final int  K_L1O2_AUGER =   16;
  public static final int  K_L1O3_AUGER =   17;
  public static final int  K_L1O4_AUGER =   18;
  public static final int  K_L1O5_AUGER =   19;
  public static final int  K_L1O6_AUGER =   20;
  public static final int  K_L1O7_AUGER =   21;
  public static final int  K_L1P1_AUGER =   22;
  public static final int  K_L1P2_AUGER =   23;
  public static final int  K_L1P3_AUGER =   24;
  public static final int  K_L1P4_AUGER =   25;
  public static final int  K_L1P5_AUGER =   26;
  public static final int  K_L1Q1_AUGER =   27;
  public static final int  K_L1Q2_AUGER =   28;
  public static final int  K_L1Q3_AUGER =   29;
  public static final int  K_L2L1_AUGER =   30;
  public static final int  K_L2L2_AUGER =   31;
  public static final int  K_L2L3_AUGER =   32;
  public static final int  K_L2M1_AUGER =   33;
  public static final int  K_L2M2_AUGER =   34;
  public static final int  K_L2M3_AUGER =   35;
  public static final int  K_L2M4_AUGER =   36;
  public static final int  K_L2M5_AUGER =   37;
  public static final int  K_L2N1_AUGER =   38;
  public static final int  K_L2N2_AUGER =   39;
  public static final int  K_L2N3_AUGER =   40;
  public static final int  K_L2N4_AUGER =   41;
  public static final int  K_L2N5_AUGER =   42;
  public static final int  K_L2N6_AUGER =   43;
  public static final int  K_L2N7_AUGER =   44;
  public static final int  K_L2O1_AUGER =   45;
  public static final int  K_L2O2_AUGER =   46;
  public static final int  K_L2O3_AUGER =   47;
  public static final int  K_L2O4_AUGER =   48;
  public static final int  K_L2O5_AUGER =   49;
  public static final int  K_L2O6_AUGER =   50;
  public static final int  K_L2O7_AUGER =   51;
  public static final int  K_L2P1_AUGER =   52;
  public static final int  K_L2P2_AUGER =   53;
  public static final int  K_L2P3_AUGER =   54;
  public static final int  K_L2P4_AUGER =   55;
  public static final int  K_L2P5_AUGER =   56;
  public static final int  K_L2Q1_AUGER =   57;
  public static final int  K_L2Q2_AUGER =   58;
  public static final int  K_L2Q3_AUGER =   59;
  public static final int  K_L3L1_AUGER =   60;
  public static final int  K_L3L2_AUGER =   61;
  public static final int  K_L3L3_AUGER =   62;
  public static final int  K_L3M1_AUGER =   63;
  public static final int  K_L3M2_AUGER =   64;
  public static final int  K_L3M3_AUGER =   65;
  public static final int  K_L3M4_AUGER =   66;
  public static final int  K_L3M5_AUGER =   67;
  public static final int  K_L3N1_AUGER =   68;
  public static final int  K_L3N2_AUGER =   69;
  public static final int  K_L3N3_AUGER =   70;
  public static final int  K_L3N4_AUGER =   71;
  public static final int  K_L3N5_AUGER =   72;
  public static final int  K_L3N6_AUGER =   73;
  public static final int  K_L3N7_AUGER =   74;
  public static final int  K_L3O1_AUGER =   75;
  public static final int  K_L3O2_AUGER =   76;
  public static final int  K_L3O3_AUGER =   77;
  public static final int  K_L3O4_AUGER =   78;
  public static final int  K_L3O5_AUGER =   79;
  public static final int  K_L3O6_AUGER =   80;
  public static final int  K_L3O7_AUGER =   81;
  public static final int  K_L3P1_AUGER =   82;
  public static final int  K_L3P2_AUGER =   83;
  public static final int  K_L3P3_AUGER =   84;
  public static final int  K_L3P4_AUGER =   85;
  public static final int  K_L3P5_AUGER =   86;
  public static final int  K_L3Q1_AUGER =   87;
  public static final int  K_L3Q2_AUGER =   88;
  public static final int  K_L3Q3_AUGER =   89;
  public static final int  K_M1L1_AUGER =   90;
  public static final int  K_M1L2_AUGER =   91;
  public static final int  K_M1L3_AUGER =   92;
  public static final int  K_M1M1_AUGER =   93;
  public static final int  K_M1M2_AUGER =   94;
  public static final int  K_M1M3_AUGER =   95;
  public static final int  K_M1M4_AUGER =   96;
  public static final int  K_M1M5_AUGER =   97;
  public static final int  K_M1N1_AUGER =   98;
  public static final int  K_M1N2_AUGER =   99;
  public static final int  K_M1N3_AUGER =  100;
  public static final int  K_M1N4_AUGER =  101;
  public static final int  K_M1N5_AUGER =  102;
  public static final int  K_M1N6_AUGER =  103;
  public static final int  K_M1N7_AUGER =  104;
  public static final int  K_M1O1_AUGER =  105;
  public static final int  K_M1O2_AUGER =  106;
  public static final int  K_M1O3_AUGER =  107;
  public static final int  K_M1O4_AUGER =  108;
  public static final int  K_M1O5_AUGER =  109;
  public static final int  K_M1O6_AUGER =  110;
  public static final int  K_M1O7_AUGER =  111;
  public static final int  K_M1P1_AUGER =  112;
  public static final int  K_M1P2_AUGER =  113;
  public static final int  K_M1P3_AUGER =  114;
  public static final int  K_M1P4_AUGER =  115;
  public static final int  K_M1P5_AUGER =  116;
  public static final int  K_M1Q1_AUGER =  117;
  public static final int  K_M1Q2_AUGER =  118;
  public static final int  K_M1Q3_AUGER =  119;
  public static final int  K_M2L1_AUGER =  120;
  public static final int  K_M2L2_AUGER =  121;
  public static final int  K_M2L3_AUGER =  122;
  public static final int  K_M2M1_AUGER =  123;
  public static final int  K_M2M2_AUGER =  124;
  public static final int  K_M2M3_AUGER =  125;
  public static final int  K_M2M4_AUGER =  126;
  public static final int  K_M2M5_AUGER =  127;
  public static final int  K_M2N1_AUGER =  128;
  public static final int  K_M2N2_AUGER =  129;
  public static final int  K_M2N3_AUGER =  130;
  public static final int  K_M2N4_AUGER =  131;
  public static final int  K_M2N5_AUGER =  132;
  public static final int  K_M2N6_AUGER =  133;
  public static final int  K_M2N7_AUGER =  134;
  public static final int  K_M2O1_AUGER =  135;
  public static final int  K_M2O2_AUGER =  136;
  public static final int  K_M2O3_AUGER =  137;
  public static final int  K_M2O4_AUGER =  138;
  public static final int  K_M2O5_AUGER =  139;
  public static final int  K_M2O6_AUGER =  140;
  public static final int  K_M2O7_AUGER =  141;
  public static final int  K_M2P1_AUGER =  142;
  public static final int  K_M2P2_AUGER =  143;
  public static final int  K_M2P3_AUGER =  144;
  public static final int  K_M2P4_AUGER =  145;
  public static final int  K_M2P5_AUGER =  146;
  public static final int  K_M2Q1_AUGER =  147;
  public static final int  K_M2Q2_AUGER =  148;
  public static final int  K_M2Q3_AUGER =  149;
  public static final int  K_M3L1_AUGER =  150;
  public static final int  K_M3L2_AUGER =  151;
  public static final int  K_M3L3_AUGER =  152;
  public static final int  K_M3M1_AUGER =  153;
  public static final int  K_M3M2_AUGER =  154;
  public static final int  K_M3M3_AUGER =  155;
  public static final int  K_M3M4_AUGER =  156;
  public static final int  K_M3M5_AUGER =  157;
  public static final int  K_M3N1_AUGER =  158;
  public static final int  K_M3N2_AUGER =  159;
  public static final int  K_M3N3_AUGER =  160;
  public static final int  K_M3N4_AUGER =  161;
  public static final int  K_M3N5_AUGER =  162;
  public static final int  K_M3N6_AUGER =  163;
  public static final int  K_M3N7_AUGER =  164;
  public static final int  K_M3O1_AUGER =  165;
  public static final int  K_M3O2_AUGER =  166;
  public static final int  K_M3O3_AUGER =  167;
  public static final int  K_M3O4_AUGER =  168;
  public static final int  K_M3O5_AUGER =  169;
  public static final int  K_M3O6_AUGER =  170;
  public static final int  K_M3O7_AUGER =  171;
  public static final int  K_M3P1_AUGER =  172;
  public static final int  K_M3P2_AUGER =  173;
  public static final int  K_M3P3_AUGER =  174;
  public static final int  K_M3P4_AUGER =  175;
  public static final int  K_M3P5_AUGER =  176;
  public static final int  K_M3Q1_AUGER =  177;
  public static final int  K_M3Q2_AUGER =  178;
  public static final int  K_M3Q3_AUGER =  179;
  public static final int  K_M4L1_AUGER =  180;
  public static final int  K_M4L2_AUGER =  181;
  public static final int  K_M4L3_AUGER =  182;
  public static final int  K_M4M1_AUGER =  183;
  public static final int  K_M4M2_AUGER =  184;
  public static final int  K_M4M3_AUGER =  185;
  public static final int  K_M4M4_AUGER =  186;
  public static final int  K_M4M5_AUGER =  187;
  public static final int  K_M4N1_AUGER =  188;
  public static final int  K_M4N2_AUGER =  189;
  public static final int  K_M4N3_AUGER =  190;
  public static final int  K_M4N4_AUGER =  191;
  public static final int  K_M4N5_AUGER =  192;
  public static final int  K_M4N6_AUGER =  193;
  public static final int  K_M4N7_AUGER =  194;
  public static final int  K_M4O1_AUGER =  195;
  public static final int  K_M4O2_AUGER =  196;
  public static final int  K_M4O3_AUGER =  197;
  public static final int  K_M4O4_AUGER =  198;
  public static final int  K_M4O5_AUGER =  199;
  public static final int  K_M4O6_AUGER =  200;
  public static final int  K_M4O7_AUGER =  201;
  public static final int  K_M4P1_AUGER =  202;
  public static final int  K_M4P2_AUGER =  203;
  public static final int  K_M4P3_AUGER =  204;
  public static final int  K_M4P4_AUGER =  205;
  public static final int  K_M4P5_AUGER =  206;
  public static final int  K_M4Q1_AUGER =  207;
  public static final int  K_M4Q2_AUGER =  208;
  public static final int  K_M4Q3_AUGER =  209;
  public static final int  K_M5L1_AUGER =  210;
  public static final int  K_M5L2_AUGER =  211;
  public static final int  K_M5L3_AUGER =  212;
  public static final int  K_M5M1_AUGER =  213;
  public static final int  K_M5M2_AUGER =  214;
  public static final int  K_M5M3_AUGER =  215;
  public static final int  K_M5M4_AUGER =  216;
  public static final int  K_M5M5_AUGER =  217;
  public static final int  K_M5N1_AUGER =  218;
  public static final int  K_M5N2_AUGER =  219;
  public static final int  K_M5N3_AUGER =  220;
  public static final int  K_M5N4_AUGER =  221;
  public static final int  K_M5N5_AUGER =  222;
  public static final int  K_M5N6_AUGER =  223;
  public static final int  K_M5N7_AUGER =  224;
  public static final int  K_M5O1_AUGER =  225;
  public static final int  K_M5O2_AUGER =  226;
  public static final int  K_M5O3_AUGER =  227;
  public static final int  K_M5O4_AUGER =  228;
  public static final int  K_M5O5_AUGER =  229;
  public static final int  K_M5O6_AUGER =  230;
  public static final int  K_M5O7_AUGER =  231;
  public static final int  K_M5P1_AUGER =  232;
  public static final int  K_M5P2_AUGER =  233;
  public static final int  K_M5P3_AUGER =  234;
  public static final int  K_M5P4_AUGER =  235;
  public static final int  K_M5P5_AUGER =  236;
  public static final int  K_M5Q1_AUGER =  237;
  public static final int  K_M5Q2_AUGER =  238;
  public static final int  K_M5Q3_AUGER =  239;
  public static final int  L1_L2L2_AUGER =  240;
  public static final int  L1_L2L3_AUGER =  241;
  public static final int  L1_L2M1_AUGER =  242;
  public static final int  L1_L2M2_AUGER =  243;
  public static final int  L1_L2M3_AUGER =  244;
  public static final int  L1_L2M4_AUGER =  245;
  public static final int  L1_L2M5_AUGER =  246;
  public static final int  L1_L2N1_AUGER =  247;
  public static final int  L1_L2N2_AUGER =  248;
  public static final int  L1_L2N3_AUGER =  249;
  public static final int  L1_L2N4_AUGER =  250;
  public static final int  L1_L2N5_AUGER =  251;
  public static final int  L1_L2N6_AUGER =  252;
  public static final int  L1_L2N7_AUGER =  253;
  public static final int  L1_L2O1_AUGER =  254;
  public static final int  L1_L2O2_AUGER =  255;
  public static final int  L1_L2O3_AUGER =  256;
  public static final int  L1_L2O4_AUGER =  257;
  public static final int  L1_L2O5_AUGER =  258;
  public static final int  L1_L2O6_AUGER =  259;
  public static final int  L1_L2O7_AUGER =  260;
  public static final int  L1_L2P1_AUGER =  261;
  public static final int  L1_L2P2_AUGER =  262;
  public static final int  L1_L2P3_AUGER =  263;
  public static final int  L1_L2P4_AUGER =  264;
  public static final int  L1_L2P5_AUGER =  265;
  public static final int  L1_L2Q1_AUGER =  266;
  public static final int  L1_L2Q2_AUGER =  267;
  public static final int  L1_L2Q3_AUGER =  268;
  public static final int  L1_L3L2_AUGER =  269;
  public static final int  L1_L3L3_AUGER =  270;
  public static final int  L1_L3M1_AUGER =  271;
  public static final int  L1_L3M2_AUGER =  272;
  public static final int  L1_L3M3_AUGER =  273;
  public static final int  L1_L3M4_AUGER =  274;
  public static final int  L1_L3M5_AUGER =  275;
  public static final int  L1_L3N1_AUGER =  276;
  public static final int  L1_L3N2_AUGER =  277;
  public static final int  L1_L3N3_AUGER =  278;
  public static final int  L1_L3N4_AUGER =  279;
  public static final int  L1_L3N5_AUGER =  280;
  public static final int  L1_L3N6_AUGER =  281;
  public static final int  L1_L3N7_AUGER =  282;
  public static final int  L1_L3O1_AUGER =  283;
  public static final int  L1_L3O2_AUGER =  284;
  public static final int  L1_L3O3_AUGER =  285;
  public static final int  L1_L3O4_AUGER =  286;
  public static final int  L1_L3O5_AUGER =  287;
  public static final int  L1_L3O6_AUGER =  288;
  public static final int  L1_L3O7_AUGER =  289;
  public static final int  L1_L3P1_AUGER =  290;
  public static final int  L1_L3P2_AUGER =  291;
  public static final int  L1_L3P3_AUGER =  292;
  public static final int  L1_L3P4_AUGER =  293;
  public static final int  L1_L3P5_AUGER =  294;
  public static final int  L1_L3Q1_AUGER =  295;
  public static final int  L1_L3Q2_AUGER =  296;
  public static final int  L1_L3Q3_AUGER =  297;
  public static final int  L1_M1L2_AUGER =  298;
  public static final int  L1_M1L3_AUGER =  299;
  public static final int  L1_M1M1_AUGER =  300;
  public static final int  L1_M1M2_AUGER =  301;
  public static final int  L1_M1M3_AUGER =  302;
  public static final int  L1_M1M4_AUGER =  303;
  public static final int  L1_M1M5_AUGER =  304;
  public static final int  L1_M1N1_AUGER =  305;
  public static final int  L1_M1N2_AUGER =  306;
  public static final int  L1_M1N3_AUGER =  307;
  public static final int  L1_M1N4_AUGER =  308;
  public static final int  L1_M1N5_AUGER =  309;
  public static final int  L1_M1N6_AUGER =  310;
  public static final int  L1_M1N7_AUGER =  311;
  public static final int  L1_M1O1_AUGER =  312;
  public static final int  L1_M1O2_AUGER =  313;
  public static final int  L1_M1O3_AUGER =  314;
  public static final int  L1_M1O4_AUGER =  315;
  public static final int  L1_M1O5_AUGER =  316;
  public static final int  L1_M1O6_AUGER =  317;
  public static final int  L1_M1O7_AUGER =  318;
  public static final int  L1_M1P1_AUGER =  319;
  public static final int  L1_M1P2_AUGER =  320;
  public static final int  L1_M1P3_AUGER =  321;
  public static final int  L1_M1P4_AUGER =  322;
  public static final int  L1_M1P5_AUGER =  323;
  public static final int  L1_M1Q1_AUGER =  324;
  public static final int  L1_M1Q2_AUGER =  325;
  public static final int  L1_M1Q3_AUGER =  326;
  public static final int  L1_M2L2_AUGER =  327;
  public static final int  L1_M2L3_AUGER =  328;
  public static final int  L1_M2M1_AUGER =  329;
  public static final int  L1_M2M2_AUGER =  330;
  public static final int  L1_M2M3_AUGER =  331;
  public static final int  L1_M2M4_AUGER =  332;
  public static final int  L1_M2M5_AUGER =  333;
  public static final int  L1_M2N1_AUGER =  334;
  public static final int  L1_M2N2_AUGER =  335;
  public static final int  L1_M2N3_AUGER =  336;
  public static final int  L1_M2N4_AUGER =  337;
  public static final int  L1_M2N5_AUGER =  338;
  public static final int  L1_M2N6_AUGER =  339;
  public static final int  L1_M2N7_AUGER =  340;
  public static final int  L1_M2O1_AUGER =  341;
  public static final int  L1_M2O2_AUGER =  342;
  public static final int  L1_M2O3_AUGER =  343;
  public static final int  L1_M2O4_AUGER =  344;
  public static final int  L1_M2O5_AUGER =  345;
  public static final int  L1_M2O6_AUGER =  346;
  public static final int  L1_M2O7_AUGER =  347;
  public static final int  L1_M2P1_AUGER =  348;
  public static final int  L1_M2P2_AUGER =  349;
  public static final int  L1_M2P3_AUGER =  350;
  public static final int  L1_M2P4_AUGER =  351;
  public static final int  L1_M2P5_AUGER =  352;
  public static final int  L1_M2Q1_AUGER =  353;
  public static final int  L1_M2Q2_AUGER =  354;
  public static final int  L1_M2Q3_AUGER =  355;
  public static final int  L1_M3L2_AUGER =  356;
  public static final int  L1_M3L3_AUGER =  357;
  public static final int  L1_M3M1_AUGER =  358;
  public static final int  L1_M3M2_AUGER =  359;
  public static final int  L1_M3M3_AUGER =  360;
  public static final int  L1_M3M4_AUGER =  361;
  public static final int  L1_M3M5_AUGER =  362;
  public static final int  L1_M3N1_AUGER =  363;
  public static final int  L1_M3N2_AUGER =  364;
  public static final int  L1_M3N3_AUGER =  365;
  public static final int  L1_M3N4_AUGER =  366;
  public static final int  L1_M3N5_AUGER =  367;
  public static final int  L1_M3N6_AUGER =  368;
  public static final int  L1_M3N7_AUGER =  369;
  public static final int  L1_M3O1_AUGER =  370;
  public static final int  L1_M3O2_AUGER =  371;
  public static final int  L1_M3O3_AUGER =  372;
  public static final int  L1_M3O4_AUGER =  373;
  public static final int  L1_M3O5_AUGER =  374;
  public static final int  L1_M3O6_AUGER =  375;
  public static final int  L1_M3O7_AUGER =  376;
  public static final int  L1_M3P1_AUGER =  377;
  public static final int  L1_M3P2_AUGER =  378;
  public static final int  L1_M3P3_AUGER =  379;
  public static final int  L1_M3P4_AUGER =  380;
  public static final int  L1_M3P5_AUGER =  381;
  public static final int  L1_M3Q1_AUGER =  382;
  public static final int  L1_M3Q2_AUGER =  383;
  public static final int  L1_M3Q3_AUGER =  384;
  public static final int  L1_M4L2_AUGER =  385;
  public static final int  L1_M4L3_AUGER =  386;
  public static final int  L1_M4M1_AUGER =  387;
  public static final int  L1_M4M2_AUGER =  388;
  public static final int  L1_M4M3_AUGER =  389;
  public static final int  L1_M4M4_AUGER =  390;
  public static final int  L1_M4M5_AUGER =  391;
  public static final int  L1_M4N1_AUGER =  392;
  public static final int  L1_M4N2_AUGER =  393;
  public static final int  L1_M4N3_AUGER =  394;
  public static final int  L1_M4N4_AUGER =  395;
  public static final int  L1_M4N5_AUGER =  396;
  public static final int  L1_M4N6_AUGER =  397;
  public static final int  L1_M4N7_AUGER =  398;
  public static final int  L1_M4O1_AUGER =  399;
  public static final int  L1_M4O2_AUGER =  400;
  public static final int  L1_M4O3_AUGER =  401;
  public static final int  L1_M4O4_AUGER =  402;
  public static final int  L1_M4O5_AUGER =  403;
  public static final int  L1_M4O6_AUGER =  404;
  public static final int  L1_M4O7_AUGER =  405;
  public static final int  L1_M4P1_AUGER =  406;
  public static final int  L1_M4P2_AUGER =  407;
  public static final int  L1_M4P3_AUGER =  408;
  public static final int  L1_M4P4_AUGER =  409;
  public static final int  L1_M4P5_AUGER =  410;
  public static final int  L1_M4Q1_AUGER =  411;
  public static final int  L1_M4Q2_AUGER =  412;
  public static final int  L1_M4Q3_AUGER =  413;
  public static final int  L1_M5L2_AUGER =  414;
  public static final int  L1_M5L3_AUGER =  415;
  public static final int  L1_M5M1_AUGER =  416;
  public static final int  L1_M5M2_AUGER =  417;
  public static final int  L1_M5M3_AUGER =  418;
  public static final int  L1_M5M4_AUGER =  419;
  public static final int  L1_M5M5_AUGER =  420;
  public static final int  L1_M5N1_AUGER =  421;
  public static final int  L1_M5N2_AUGER =  422;
  public static final int  L1_M5N3_AUGER =  423;
  public static final int  L1_M5N4_AUGER =  424;
  public static final int  L1_M5N5_AUGER =  425;
  public static final int  L1_M5N6_AUGER =  426;
  public static final int  L1_M5N7_AUGER =  427;
  public static final int  L1_M5O1_AUGER =  428;
  public static final int  L1_M5O2_AUGER =  429;
  public static final int  L1_M5O3_AUGER =  430;
  public static final int  L1_M5O4_AUGER =  431;
  public static final int  L1_M5O5_AUGER =  432;
  public static final int  L1_M5O6_AUGER =  433;
  public static final int  L1_M5O7_AUGER =  434;
  public static final int  L1_M5P1_AUGER =  435;
  public static final int  L1_M5P2_AUGER =  436;
  public static final int  L1_M5P3_AUGER =  437;
  public static final int  L1_M5P4_AUGER =  438;
  public static final int  L1_M5P5_AUGER =  439;
  public static final int  L1_M5Q1_AUGER =  440;
  public static final int  L1_M5Q2_AUGER =  441;
  public static final int  L1_M5Q3_AUGER =  442;
  public static final int  L2_L3L3_AUGER =  443;
  public static final int  L2_L3M1_AUGER =  444;
  public static final int  L2_L3M2_AUGER =  445;
  public static final int  L2_L3M3_AUGER =  446;
  public static final int  L2_L3M4_AUGER =  447;
  public static final int  L2_L3M5_AUGER =  448;
  public static final int  L2_L3N1_AUGER =  449;
  public static final int  L2_L3N2_AUGER =  450;
  public static final int  L2_L3N3_AUGER =  451;
  public static final int  L2_L3N4_AUGER =  452;
  public static final int  L2_L3N5_AUGER =  453;
  public static final int  L2_L3N6_AUGER =  454;
  public static final int  L2_L3N7_AUGER =  455;
  public static final int  L2_L3O1_AUGER =  456;
  public static final int  L2_L3O2_AUGER =  457;
  public static final int  L2_L3O3_AUGER =  458;
  public static final int  L2_L3O4_AUGER =  459;
  public static final int  L2_L3O5_AUGER =  460;
  public static final int  L2_L3O6_AUGER =  461;
  public static final int  L2_L3O7_AUGER =  462;
  public static final int  L2_L3P1_AUGER =  463;
  public static final int  L2_L3P2_AUGER =  464;
  public static final int  L2_L3P3_AUGER =  465;
  public static final int  L2_L3P4_AUGER =  466;
  public static final int  L2_L3P5_AUGER =  467;
  public static final int  L2_L3Q1_AUGER =  468;
  public static final int  L2_L3Q2_AUGER =  469;
  public static final int  L2_L3Q3_AUGER =  470;
  public static final int  L2_M1L3_AUGER =  471;
  public static final int  L2_M1M1_AUGER =  472;
  public static final int  L2_M1M2_AUGER =  473;
  public static final int  L2_M1M3_AUGER =  474;
  public static final int  L2_M1M4_AUGER =  475;
  public static final int  L2_M1M5_AUGER =  476;
  public static final int  L2_M1N1_AUGER =  477;
  public static final int  L2_M1N2_AUGER =  478;
  public static final int  L2_M1N3_AUGER =  479;
  public static final int  L2_M1N4_AUGER =  480;
  public static final int  L2_M1N5_AUGER =  481;
  public static final int  L2_M1N6_AUGER =  482;
  public static final int  L2_M1N7_AUGER =  483;
  public static final int  L2_M1O1_AUGER =  484;
  public static final int  L2_M1O2_AUGER =  485;
  public static final int  L2_M1O3_AUGER =  486;
  public static final int  L2_M1O4_AUGER =  487;
  public static final int  L2_M1O5_AUGER =  488;
  public static final int  L2_M1O6_AUGER =  489;
  public static final int  L2_M1O7_AUGER =  490;
  public static final int  L2_M1P1_AUGER =  491;
  public static final int  L2_M1P2_AUGER =  492;
  public static final int  L2_M1P3_AUGER =  493;
  public static final int  L2_M1P4_AUGER =  494;
  public static final int  L2_M1P5_AUGER =  495;
  public static final int  L2_M1Q1_AUGER =  496;
  public static final int  L2_M1Q2_AUGER =  497;
  public static final int  L2_M1Q3_AUGER =  498;
  public static final int  L2_M2L3_AUGER =  499;
  public static final int  L2_M2M1_AUGER =  500;
  public static final int  L2_M2M2_AUGER =  501;
  public static final int  L2_M2M3_AUGER =  502;
  public static final int  L2_M2M4_AUGER =  503;
  public static final int  L2_M2M5_AUGER =  504;
  public static final int  L2_M2N1_AUGER =  505;
  public static final int  L2_M2N2_AUGER =  506;
  public static final int  L2_M2N3_AUGER =  507;
  public static final int  L2_M2N4_AUGER =  508;
  public static final int  L2_M2N5_AUGER =  509;
  public static final int  L2_M2N6_AUGER =  510;
  public static final int  L2_M2N7_AUGER =  511;
  public static final int  L2_M2O1_AUGER =  512;
  public static final int  L2_M2O2_AUGER =  513;
  public static final int  L2_M2O3_AUGER =  514;
  public static final int  L2_M2O4_AUGER =  515;
  public static final int  L2_M2O5_AUGER =  516;
  public static final int  L2_M2O6_AUGER =  517;
  public static final int  L2_M2O7_AUGER =  518;
  public static final int  L2_M2P1_AUGER =  519;
  public static final int  L2_M2P2_AUGER =  520;
  public static final int  L2_M2P3_AUGER =  521;
  public static final int  L2_M2P4_AUGER =  522;
  public static final int  L2_M2P5_AUGER =  523;
  public static final int  L2_M2Q1_AUGER =  524;
  public static final int  L2_M2Q2_AUGER =  525;
  public static final int  L2_M2Q3_AUGER =  526;
  public static final int  L2_M3L3_AUGER =  527;
  public static final int  L2_M3M1_AUGER =  528;
  public static final int  L2_M3M2_AUGER =  529;
  public static final int  L2_M3M3_AUGER =  530;
  public static final int  L2_M3M4_AUGER =  531;
  public static final int  L2_M3M5_AUGER =  532;
  public static final int  L2_M3N1_AUGER =  533;
  public static final int  L2_M3N2_AUGER =  534;
  public static final int  L2_M3N3_AUGER =  535;
  public static final int  L2_M3N4_AUGER =  536;
  public static final int  L2_M3N5_AUGER =  537;
  public static final int  L2_M3N6_AUGER =  538;
  public static final int  L2_M3N7_AUGER =  539;
  public static final int  L2_M3O1_AUGER =  540;
  public static final int  L2_M3O2_AUGER =  541;
  public static final int  L2_M3O3_AUGER =  542;
  public static final int  L2_M3O4_AUGER =  543;
  public static final int  L2_M3O5_AUGER =  544;
  public static final int  L2_M3O6_AUGER =  545;
  public static final int  L2_M3O7_AUGER =  546;
  public static final int  L2_M3P1_AUGER =  547;
  public static final int  L2_M3P2_AUGER =  548;
  public static final int  L2_M3P3_AUGER =  549;
  public static final int  L2_M3P4_AUGER =  550;
  public static final int  L2_M3P5_AUGER =  551;
  public static final int  L2_M3Q1_AUGER =  552;
  public static final int  L2_M3Q2_AUGER =  553;
  public static final int  L2_M3Q3_AUGER =  554;
  public static final int  L2_M4L3_AUGER =  555;
  public static final int  L2_M4M1_AUGER =  556;
  public static final int  L2_M4M2_AUGER =  557;
  public static final int  L2_M4M3_AUGER =  558;
  public static final int  L2_M4M4_AUGER =  559;
  public static final int  L2_M4M5_AUGER =  560;
  public static final int  L2_M4N1_AUGER =  561;
  public static final int  L2_M4N2_AUGER =  562;
  public static final int  L2_M4N3_AUGER =  563;
  public static final int  L2_M4N4_AUGER =  564;
  public static final int  L2_M4N5_AUGER =  565;
  public static final int  L2_M4N6_AUGER =  566;
  public static final int  L2_M4N7_AUGER =  567;
  public static final int  L2_M4O1_AUGER =  568;
  public static final int  L2_M4O2_AUGER =  569;
  public static final int  L2_M4O3_AUGER =  570;
  public static final int  L2_M4O4_AUGER =  571;
  public static final int  L2_M4O5_AUGER =  572;
  public static final int  L2_M4O6_AUGER =  573;
  public static final int  L2_M4O7_AUGER =  574;
  public static final int  L2_M4P1_AUGER =  575;
  public static final int  L2_M4P2_AUGER =  576;
  public static final int  L2_M4P3_AUGER =  577;
  public static final int  L2_M4P4_AUGER =  578;
  public static final int  L2_M4P5_AUGER =  579;
  public static final int  L2_M4Q1_AUGER =  580;
  public static final int  L2_M4Q2_AUGER =  581;
  public static final int  L2_M4Q3_AUGER =  582;
  public static final int  L2_M5L3_AUGER =  583;
  public static final int  L2_M5M1_AUGER =  584;
  public static final int  L2_M5M2_AUGER =  585;
  public static final int  L2_M5M3_AUGER =  586;
  public static final int  L2_M5M4_AUGER =  587;
  public static final int  L2_M5M5_AUGER =  588;
  public static final int  L2_M5N1_AUGER =  589;
  public static final int  L2_M5N2_AUGER =  590;
  public static final int  L2_M5N3_AUGER =  591;
  public static final int  L2_M5N4_AUGER =  592;
  public static final int  L2_M5N5_AUGER =  593;
  public static final int  L2_M5N6_AUGER =  594;
  public static final int  L2_M5N7_AUGER =  595;
  public static final int  L2_M5O1_AUGER =  596;
  public static final int  L2_M5O2_AUGER =  597;
  public static final int  L2_M5O3_AUGER =  598;
  public static final int  L2_M5O4_AUGER =  599;
  public static final int  L2_M5O5_AUGER =  600;
  public static final int  L2_M5O6_AUGER =  601;
  public static final int  L2_M5O7_AUGER =  602;
  public static final int  L2_M5P1_AUGER =  603;
  public static final int  L2_M5P2_AUGER =  604;
  public static final int  L2_M5P3_AUGER =  605;
  public static final int  L2_M5P4_AUGER =  606;
  public static final int  L2_M5P5_AUGER =  607;
  public static final int  L2_M5Q1_AUGER =  608;
  public static final int  L2_M5Q2_AUGER =  609;
  public static final int  L2_M5Q3_AUGER =  610;
  public static final int  L3_M1M1_AUGER =  611;
  public static final int  L3_M1M2_AUGER =  612;
  public static final int  L3_M1M3_AUGER =  613;
  public static final int  L3_M1M4_AUGER =  614;
  public static final int  L3_M1M5_AUGER =  615;
  public static final int  L3_M1N1_AUGER =  616;
  public static final int  L3_M1N2_AUGER =  617;
  public static final int  L3_M1N3_AUGER =  618;
  public static final int  L3_M1N4_AUGER =  619;
  public static final int  L3_M1N5_AUGER =  620;
  public static final int  L3_M1N6_AUGER =  621;
  public static final int  L3_M1N7_AUGER =  622;
  public static final int  L3_M1O1_AUGER =  623;
  public static final int  L3_M1O2_AUGER =  624;
  public static final int  L3_M1O3_AUGER =  625;
  public static final int  L3_M1O4_AUGER =  626;
  public static final int  L3_M1O5_AUGER =  627;
  public static final int  L3_M1O6_AUGER =  628;
  public static final int  L3_M1O7_AUGER =  629;
  public static final int  L3_M1P1_AUGER =  630;
  public static final int  L3_M1P2_AUGER =  631;
  public static final int  L3_M1P3_AUGER =  632;
  public static final int  L3_M1P4_AUGER =  633;
  public static final int  L3_M1P5_AUGER =  634;
  public static final int  L3_M1Q1_AUGER =  635;
  public static final int  L3_M1Q2_AUGER =  636;
  public static final int  L3_M1Q3_AUGER =  637;
  public static final int  L3_M2M1_AUGER =  638;
  public static final int  L3_M2M2_AUGER =  639;
  public static final int  L3_M2M3_AUGER =  640;
  public static final int  L3_M2M4_AUGER =  641;
  public static final int  L3_M2M5_AUGER =  642;
  public static final int  L3_M2N1_AUGER =  643;
  public static final int  L3_M2N2_AUGER =  644;
  public static final int  L3_M2N3_AUGER =  645;
  public static final int  L3_M2N4_AUGER =  646;
  public static final int  L3_M2N5_AUGER =  647;
  public static final int  L3_M2N6_AUGER =  648;
  public static final int  L3_M2N7_AUGER =  649;
  public static final int  L3_M2O1_AUGER =  650;
  public static final int  L3_M2O2_AUGER =  651;
  public static final int  L3_M2O3_AUGER =  652;
  public static final int  L3_M2O4_AUGER =  653;
  public static final int  L3_M2O5_AUGER =  654;
  public static final int  L3_M2O6_AUGER =  655;
  public static final int  L3_M2O7_AUGER =  656;
  public static final int  L3_M2P1_AUGER =  657;
  public static final int  L3_M2P2_AUGER =  658;
  public static final int  L3_M2P3_AUGER =  659;
  public static final int  L3_M2P4_AUGER =  660;
  public static final int  L3_M2P5_AUGER =  661;
  public static final int  L3_M2Q1_AUGER =  662;
  public static final int  L3_M2Q2_AUGER =  663;
  public static final int  L3_M2Q3_AUGER =  664;
  public static final int  L3_M3M1_AUGER =  665;
  public static final int  L3_M3M2_AUGER =  666;
  public static final int  L3_M3M3_AUGER =  667;
  public static final int  L3_M3M4_AUGER =  668;
  public static final int  L3_M3M5_AUGER =  669;
  public static final int  L3_M3N1_AUGER =  670;
  public static final int  L3_M3N2_AUGER =  671;
  public static final int  L3_M3N3_AUGER =  672;
  public static final int  L3_M3N4_AUGER =  673;
  public static final int  L3_M3N5_AUGER =  674;
  public static final int  L3_M3N6_AUGER =  675;
  public static final int  L3_M3N7_AUGER =  676;
  public static final int  L3_M3O1_AUGER =  677;
  public static final int  L3_M3O2_AUGER =  678;
  public static final int  L3_M3O3_AUGER =  679;
  public static final int  L3_M3O4_AUGER =  680;
  public static final int  L3_M3O5_AUGER =  681;
  public static final int  L3_M3O6_AUGER =  682;
  public static final int  L3_M3O7_AUGER =  683;
  public static final int  L3_M3P1_AUGER =  684;
  public static final int  L3_M3P2_AUGER =  685;
  public static final int  L3_M3P3_AUGER =  686;
  public static final int  L3_M3P4_AUGER =  687;
  public static final int  L3_M3P5_AUGER =  688;
  public static final int  L3_M3Q1_AUGER =  689;
  public static final int  L3_M3Q2_AUGER =  690;
  public static final int  L3_M3Q3_AUGER =  691;
  public static final int  L3_M4M1_AUGER =  692;
  public static final int  L3_M4M2_AUGER =  693;
  public static final int  L3_M4M3_AUGER =  694;
  public static final int  L3_M4M4_AUGER =  695;
  public static final int  L3_M4M5_AUGER =  696;
  public static final int  L3_M4N1_AUGER =  697;
  public static final int  L3_M4N2_AUGER =  698;
  public static final int  L3_M4N3_AUGER =  699;
  public static final int  L3_M4N4_AUGER =  700;
  public static final int  L3_M4N5_AUGER =  701;
  public static final int  L3_M4N6_AUGER =  702;
  public static final int  L3_M4N7_AUGER =  703;
  public static final int  L3_M4O1_AUGER =  704;
  public static final int  L3_M4O2_AUGER =  705;
  public static final int  L3_M4O3_AUGER =  706;
  public static final int  L3_M4O4_AUGER =  707;
  public static final int  L3_M4O5_AUGER =  708;
  public static final int  L3_M4O6_AUGER =  709;
  public static final int  L3_M4O7_AUGER =  710;
  public static final int  L3_M4P1_AUGER =  711;
  public static final int  L3_M4P2_AUGER =  712;
  public static final int  L3_M4P3_AUGER =  713;
  public static final int  L3_M4P4_AUGER =  714;
  public static final int  L3_M4P5_AUGER =  715;
  public static final int  L3_M4Q1_AUGER =  716;
  public static final int  L3_M4Q2_AUGER =  717;
  public static final int  L3_M4Q3_AUGER =  718;
  public static final int  L3_M5M1_AUGER =  719;
  public static final int  L3_M5M2_AUGER =  720;
  public static final int  L3_M5M3_AUGER =  721;
  public static final int  L3_M5M4_AUGER =  722;
  public static final int  L3_M5M5_AUGER =  723;
  public static final int  L3_M5N1_AUGER =  724;
  public static final int  L3_M5N2_AUGER =  725;
  public static final int  L3_M5N3_AUGER =  726;
  public static final int  L3_M5N4_AUGER =  727;
  public static final int  L3_M5N5_AUGER =  728;
  public static final int  L3_M5N6_AUGER =  729;
  public static final int  L3_M5N7_AUGER =  730;
  public static final int  L3_M5O1_AUGER =  731;
  public static final int  L3_M5O2_AUGER =  732;
  public static final int  L3_M5O3_AUGER =  733;
  public static final int  L3_M5O4_AUGER =  734;
  public static final int  L3_M5O5_AUGER =  735;
  public static final int  L3_M5O6_AUGER =  736;
  public static final int  L3_M5O7_AUGER =  737;
  public static final int  L3_M5P1_AUGER =  738;
  public static final int  L3_M5P2_AUGER =  739;
  public static final int  L3_M5P3_AUGER =  740;
  public static final int  L3_M5P4_AUGER =  741;
  public static final int  L3_M5P5_AUGER =  742;
  public static final int  L3_M5Q1_AUGER =  743;
  public static final int  L3_M5Q2_AUGER =  744;
  public static final int  L3_M5Q3_AUGER =  745;
  public static final int  M1_M2M2_AUGER =  746;
  public static final int  M1_M2M3_AUGER =  747;
  public static final int  M1_M2M4_AUGER =  748;
  public static final int  M1_M2M5_AUGER =  749;
  public static final int  M1_M2N1_AUGER =  750;
  public static final int  M1_M2N2_AUGER =  751;
  public static final int  M1_M2N3_AUGER =  752;
  public static final int  M1_M2N4_AUGER =  753;
  public static final int  M1_M2N5_AUGER =  754;
  public static final int  M1_M2N6_AUGER =  755;
  public static final int  M1_M2N7_AUGER =  756;
  public static final int  M1_M2O1_AUGER =  757;
  public static final int  M1_M2O2_AUGER =  758;
  public static final int  M1_M2O3_AUGER =  759;
  public static final int  M1_M2O4_AUGER =  760;
  public static final int  M1_M2O5_AUGER =  761;
  public static final int  M1_M2O6_AUGER =  762;
  public static final int  M1_M2O7_AUGER =  763;
  public static final int  M1_M2P1_AUGER =  764;
  public static final int  M1_M2P2_AUGER =  765;
  public static final int  M1_M2P3_AUGER =  766;
  public static final int  M1_M2P4_AUGER =  767;
  public static final int  M1_M2P5_AUGER =  768;
  public static final int  M1_M2Q1_AUGER =  769;
  public static final int  M1_M2Q2_AUGER =  770;
  public static final int  M1_M2Q3_AUGER =  771;
  public static final int  M1_M3M2_AUGER =  772;
  public static final int  M1_M3M3_AUGER =  773;
  public static final int  M1_M3M4_AUGER =  774;
  public static final int  M1_M3M5_AUGER =  775;
  public static final int  M1_M3N1_AUGER =  776;
  public static final int  M1_M3N2_AUGER =  777;
  public static final int  M1_M3N3_AUGER =  778;
  public static final int  M1_M3N4_AUGER =  779;
  public static final int  M1_M3N5_AUGER =  780;
  public static final int  M1_M3N6_AUGER =  781;
  public static final int  M1_M3N7_AUGER =  782;
  public static final int  M1_M3O1_AUGER =  783;
  public static final int  M1_M3O2_AUGER =  784;
  public static final int  M1_M3O3_AUGER =  785;
  public static final int  M1_M3O4_AUGER =  786;
  public static final int  M1_M3O5_AUGER =  787;
  public static final int  M1_M3O6_AUGER =  788;
  public static final int  M1_M3O7_AUGER =  789;
  public static final int  M1_M3P1_AUGER =  790;
  public static final int  M1_M3P2_AUGER =  791;
  public static final int  M1_M3P3_AUGER =  792;
  public static final int  M1_M3P4_AUGER =  793;
  public static final int  M1_M3P5_AUGER =  794;
  public static final int  M1_M3Q1_AUGER =  795;
  public static final int  M1_M3Q2_AUGER =  796;
  public static final int  M1_M3Q3_AUGER =  797;
  public static final int  M1_M4M2_AUGER =  798;
  public static final int  M1_M4M3_AUGER =  799;
  public static final int  M1_M4M4_AUGER =  800;
  public static final int  M1_M4M5_AUGER =  801;
  public static final int  M1_M4N1_AUGER =  802;
  public static final int  M1_M4N2_AUGER =  803;
  public static final int  M1_M4N3_AUGER =  804;
  public static final int  M1_M4N4_AUGER =  805;
  public static final int  M1_M4N5_AUGER =  806;
  public static final int  M1_M4N6_AUGER =  807;
  public static final int  M1_M4N7_AUGER =  808;
  public static final int  M1_M4O1_AUGER =  809;
  public static final int  M1_M4O2_AUGER =  810;
  public static final int  M1_M4O3_AUGER =  811;
  public static final int  M1_M4O4_AUGER =  812;
  public static final int  M1_M4O5_AUGER =  813;
  public static final int  M1_M4O6_AUGER =  814;
  public static final int  M1_M4O7_AUGER =  815;
  public static final int  M1_M4P1_AUGER =  816;
  public static final int  M1_M4P2_AUGER =  817;
  public static final int  M1_M4P3_AUGER =  818;
  public static final int  M1_M4P4_AUGER =  819;
  public static final int  M1_M4P5_AUGER =  820;
  public static final int  M1_M4Q1_AUGER =  821;
  public static final int  M1_M4Q2_AUGER =  822;
  public static final int  M1_M4Q3_AUGER =  823;
  public static final int  M1_M5M2_AUGER =  824;
  public static final int  M1_M5M3_AUGER =  825;
  public static final int  M1_M5M4_AUGER =  826;
  public static final int  M1_M5M5_AUGER =  827;
  public static final int  M1_M5N1_AUGER =  828;
  public static final int  M1_M5N2_AUGER =  829;
  public static final int  M1_M5N3_AUGER =  830;
  public static final int  M1_M5N4_AUGER =  831;
  public static final int  M1_M5N5_AUGER =  832;
  public static final int  M1_M5N6_AUGER =  833;
  public static final int  M1_M5N7_AUGER =  834;
  public static final int  M1_M5O1_AUGER =  835;
  public static final int  M1_M5O2_AUGER =  836;
  public static final int  M1_M5O3_AUGER =  837;
  public static final int  M1_M5O4_AUGER =  838;
  public static final int  M1_M5O5_AUGER =  839;
  public static final int  M1_M5O6_AUGER =  840;
  public static final int  M1_M5O7_AUGER =  841;
  public static final int  M1_M5P1_AUGER =  842;
  public static final int  M1_M5P2_AUGER =  843;
  public static final int  M1_M5P3_AUGER =  844;
  public static final int  M1_M5P4_AUGER =  845;
  public static final int  M1_M5P5_AUGER =  846;
  public static final int  M1_M5Q1_AUGER =  847;
  public static final int  M1_M5Q2_AUGER =  848;
  public static final int  M1_M5Q3_AUGER =  849;
  public static final int  M2_M3M3_AUGER =  850;
  public static final int  M2_M3M4_AUGER =  851;
  public static final int  M2_M3M5_AUGER =  852;
  public static final int  M2_M3N1_AUGER =  853;
  public static final int  M2_M3N2_AUGER =  854;
  public static final int  M2_M3N3_AUGER =  855;
  public static final int  M2_M3N4_AUGER =  856;
  public static final int  M2_M3N5_AUGER =  857;
  public static final int  M2_M3N6_AUGER =  858;
  public static final int  M2_M3N7_AUGER =  859;
  public static final int  M2_M3O1_AUGER =  860;
  public static final int  M2_M3O2_AUGER =  861;
  public static final int  M2_M3O3_AUGER =  862;
  public static final int  M2_M3O4_AUGER =  863;
  public static final int  M2_M3O5_AUGER =  864;
  public static final int  M2_M3O6_AUGER =  865;
  public static final int  M2_M3O7_AUGER =  866;
  public static final int  M2_M3P1_AUGER =  867;
  public static final int  M2_M3P2_AUGER =  868;
  public static final int  M2_M3P3_AUGER =  869;
  public static final int  M2_M3P4_AUGER =  870;
  public static final int  M2_M3P5_AUGER =  871;
  public static final int  M2_M3Q1_AUGER =  872;
  public static final int  M2_M3Q2_AUGER =  873;
  public static final int  M2_M3Q3_AUGER =  874;
  public static final int  M2_M4M3_AUGER =  875;
  public static final int  M2_M4M4_AUGER =  876;
  public static final int  M2_M4M5_AUGER =  877;
  public static final int  M2_M4N1_AUGER =  878;
  public static final int  M2_M4N2_AUGER =  879;
  public static final int  M2_M4N3_AUGER =  880;
  public static final int  M2_M4N4_AUGER =  881;
  public static final int  M2_M4N5_AUGER =  882;
  public static final int  M2_M4N6_AUGER =  883;
  public static final int  M2_M4N7_AUGER =  884;
  public static final int  M2_M4O1_AUGER =  885;
  public static final int  M2_M4O2_AUGER =  886;
  public static final int  M2_M4O3_AUGER =  887;
  public static final int  M2_M4O4_AUGER =  888;
  public static final int  M2_M4O5_AUGER =  889;
  public static final int  M2_M4O6_AUGER =  890;
  public static final int  M2_M4O7_AUGER =  891;
  public static final int  M2_M4P1_AUGER =  892;
  public static final int  M2_M4P2_AUGER =  893;
  public static final int  M2_M4P3_AUGER =  894;
  public static final int  M2_M4P4_AUGER =  895;
  public static final int  M2_M4P5_AUGER =  896;
  public static final int  M2_M4Q1_AUGER =  897;
  public static final int  M2_M4Q2_AUGER =  898;
  public static final int  M2_M4Q3_AUGER =  899;
  public static final int  M2_M5M3_AUGER =  900;
  public static final int  M2_M5M4_AUGER =  901;
  public static final int  M2_M5M5_AUGER =  902;
  public static final int  M2_M5N1_AUGER =  903;
  public static final int  M2_M5N2_AUGER =  904;
  public static final int  M2_M5N3_AUGER =  905;
  public static final int  M2_M5N4_AUGER =  906;
  public static final int  M2_M5N5_AUGER =  907;
  public static final int  M2_M5N6_AUGER =  908;
  public static final int  M2_M5N7_AUGER =  909;
  public static final int  M2_M5O1_AUGER =  910;
  public static final int  M2_M5O2_AUGER =  911;
  public static final int  M2_M5O3_AUGER =  912;
  public static final int  M2_M5O4_AUGER =  913;
  public static final int  M2_M5O5_AUGER =  914;
  public static final int  M2_M5O6_AUGER =  915;
  public static final int  M2_M5O7_AUGER =  916;
  public static final int  M2_M5P1_AUGER =  917;
  public static final int  M2_M5P2_AUGER =  918;
  public static final int  M2_M5P3_AUGER =  919;
  public static final int  M2_M5P4_AUGER =  920;
  public static final int  M2_M5P5_AUGER =  921;
  public static final int  M2_M5Q1_AUGER =  922;
  public static final int  M2_M5Q2_AUGER =  923;
  public static final int  M2_M5Q3_AUGER =  924;
  public static final int  M3_M4M4_AUGER =  925;
  public static final int  M3_M4M5_AUGER =  926;
  public static final int  M3_M4N1_AUGER =  927;
  public static final int  M3_M4N2_AUGER =  928;
  public static final int  M3_M4N3_AUGER =  929;
  public static final int  M3_M4N4_AUGER =  930;
  public static final int  M3_M4N5_AUGER =  931;
  public static final int  M3_M4N6_AUGER =  932;
  public static final int  M3_M4N7_AUGER =  933;
  public static final int  M3_M4O1_AUGER =  934;
  public static final int  M3_M4O2_AUGER =  935;
  public static final int  M3_M4O3_AUGER =  936;
  public static final int  M3_M4O4_AUGER =  937;
  public static final int  M3_M4O5_AUGER =  938;
  public static final int  M3_M4O6_AUGER =  939;
  public static final int  M3_M4O7_AUGER =  940;
  public static final int  M3_M4P1_AUGER =  941;
  public static final int  M3_M4P2_AUGER =  942;
  public static final int  M3_M4P3_AUGER =  943;
  public static final int  M3_M4P4_AUGER =  944;
  public static final int  M3_M4P5_AUGER =  945;
  public static final int  M3_M4Q1_AUGER =  946;
  public static final int  M3_M4Q2_AUGER =  947;
  public static final int  M3_M4Q3_AUGER =  948;
  public static final int  M3_M5M4_AUGER =  949;
  public static final int  M3_M5M5_AUGER =  950;
  public static final int  M3_M5N1_AUGER =  951;
  public static final int  M3_M5N2_AUGER =  952;
  public static final int  M3_M5N3_AUGER =  953;
  public static final int  M3_M5N4_AUGER =  954;
  public static final int  M3_M5N5_AUGER =  955;
  public static final int  M3_M5N6_AUGER =  956;
  public static final int  M3_M5N7_AUGER =  957;
  public static final int  M3_M5O1_AUGER =  958;
  public static final int  M3_M5O2_AUGER =  959;
  public static final int  M3_M5O3_AUGER =  960;
  public static final int  M3_M5O4_AUGER =  961;
  public static final int  M3_M5O5_AUGER =  962;
  public static final int  M3_M5O6_AUGER =  963;
  public static final int  M3_M5O7_AUGER =  964;
  public static final int  M3_M5P1_AUGER =  965;
  public static final int  M3_M5P2_AUGER =  966;
  public static final int  M3_M5P3_AUGER =  967;
  public static final int  M3_M5P4_AUGER =  968;
  public static final int  M3_M5P5_AUGER =  969;
  public static final int  M3_M5Q1_AUGER =  970;
  public static final int  M3_M5Q2_AUGER =  971;
  public static final int  M3_M5Q3_AUGER =  972;
  public static final int  M4_M5M5_AUGER =  973;
  public static final int  M4_M5N1_AUGER =  974;
  public static final int  M4_M5N2_AUGER =  975;
  public static final int  M4_M5N3_AUGER =  976;
  public static final int  M4_M5N4_AUGER =  977;
  public static final int  M4_M5N5_AUGER =  978;
  public static final int  M4_M5N6_AUGER =  979;
  public static final int  M4_M5N7_AUGER =  980;
  public static final int  M4_M5O1_AUGER =  981;
  public static final int  M4_M5O2_AUGER =  982;
  public static final int  M4_M5O3_AUGER =  983;
  public static final int  M4_M5O4_AUGER =  984;
  public static final int  M4_M5O5_AUGER =  985;
  public static final int  M4_M5O6_AUGER =  986;
  public static final int  M4_M5O7_AUGER =  987;
  public static final int  M4_M5P1_AUGER =  988;
  public static final int  M4_M5P2_AUGER =  989;
  public static final int  M4_M5P3_AUGER =  990;
  public static final int  M4_M5P4_AUGER =  991;
  public static final int  M4_M5P5_AUGER =  992;
  public static final int  M4_M5Q1_AUGER =  993;
  public static final int  M4_M5Q2_AUGER =  994;
  public static final int  M4_M5Q3_AUGER =  995;

  public static final int NIST_COMPOUND_A_150_TISSUE_EQUIVALENT_PLASTIC = 0;
  public static final int NIST_COMPOUND_ACETONE = 1;
  public static final int NIST_COMPOUND_ACETYLENE = 2;
  public static final int NIST_COMPOUND_ADENINE = 3;
  public static final int NIST_COMPOUND_ADIPOSE_TISSUE_ICRP = 4;
  public static final int NIST_COMPOUND_AIR_DRY_NEAR_SEA_LEVEL = 5;
  public static final int NIST_COMPOUND_ALANINE = 6;
  public static final int NIST_COMPOUND_ALUMINUM_OXIDE = 7;
  public static final int NIST_COMPOUND_AMBER = 8;
  public static final int NIST_COMPOUND_AMMONIA = 9;
  public static final int NIST_COMPOUND_ANILINE = 10;
  public static final int NIST_COMPOUND_ANTHRACENE = 11;
  public static final int NIST_COMPOUND_B_100_BONE_EQUIVALENT_PLASTIC = 12;
  public static final int NIST_COMPOUND_BAKELITE = 13;
  public static final int NIST_COMPOUND_BARIUM_FLUORIDE = 14;
  public static final int NIST_COMPOUND_BARIUM_SULFATE = 15;
  public static final int NIST_COMPOUND_BENZENE = 16;
  public static final int NIST_COMPOUND_BERYLLIUM_OXIDE = 17;
  public static final int NIST_COMPOUND_BISMUTH_GERMANIUM_OXIDE = 18;
  public static final int NIST_COMPOUND_BLOOD_ICRP = 19;
  public static final int NIST_COMPOUND_BONE_COMPACT_ICRU = 20;
  public static final int NIST_COMPOUND_BONE_CORTICAL_ICRP = 21;
  public static final int NIST_COMPOUND_BORON_CARBIDE = 22;
  public static final int NIST_COMPOUND_BORON_OXIDE = 23;
  public static final int NIST_COMPOUND_BRAIN_ICRP = 24;
  public static final int NIST_COMPOUND_BUTANE = 25;
  public static final int NIST_COMPOUND_N_BUTYL_ALCOHOL = 26;
  public static final int NIST_COMPOUND_C_552_AIR_EQUIVALENT_PLASTIC = 27;
  public static final int NIST_COMPOUND_CADMIUM_TELLURIDE = 28;
  public static final int NIST_COMPOUND_CADMIUM_TUNGSTATE = 29;
  public static final int NIST_COMPOUND_CALCIUM_CARBONATE = 30;
  public static final int NIST_COMPOUND_CALCIUM_FLUORIDE = 31;
  public static final int NIST_COMPOUND_CALCIUM_OXIDE = 32;
  public static final int NIST_COMPOUND_CALCIUM_SULFATE = 33;
  public static final int NIST_COMPOUND_CALCIUM_TUNGSTATE = 34;
  public static final int NIST_COMPOUND_CARBON_DIOXIDE = 35;
  public static final int NIST_COMPOUND_CARBON_TETRACHLORIDE = 36;
  public static final int NIST_COMPOUND_CELLULOSE_ACETATE_CELLOPHANE = 37;
  public static final int NIST_COMPOUND_CELLULOSE_ACETATE_BUTYRATE = 38;
  public static final int NIST_COMPOUND_CELLULOSE_NITRATE = 39;
  public static final int NIST_COMPOUND_CERIC_SULFATE_DOSIMETER_SOLUTION = 40;
  public static final int NIST_COMPOUND_CESIUM_FLUORIDE = 41;
  public static final int NIST_COMPOUND_CESIUM_IODIDE = 42;
  public static final int NIST_COMPOUND_CHLOROBENZENE = 43;
  public static final int NIST_COMPOUND_CHLOROFORM = 44;
  public static final int NIST_COMPOUND_CONCRETE_PORTLAND = 45;
  public static final int NIST_COMPOUND_CYCLOHEXANE = 46;
  public static final int NIST_COMPOUND_12_DDIHLOROBENZENE = 47;
  public static final int NIST_COMPOUND_DICHLORODIETHYL_ETHER = 48;
  public static final int NIST_COMPOUND_12_DICHLOROETHANE = 49;
  public static final int NIST_COMPOUND_DIETHYL_ETHER = 50;
  public static final int NIST_COMPOUND_NN_DIMETHYL_FORMAMIDE = 51;
  public static final int NIST_COMPOUND_DIMETHYL_SULFOXIDE = 52;
  public static final int NIST_COMPOUND_ETHANE = 53;
  public static final int NIST_COMPOUND_ETHYL_ALCOHOL = 54;
  public static final int NIST_COMPOUND_ETHYL_CELLULOSE = 55;
  public static final int NIST_COMPOUND_ETHYLENE = 56;
  public static final int NIST_COMPOUND_EYE_LENS_ICRP = 57;
  public static final int NIST_COMPOUND_FERRIC_OXIDE = 58;
  public static final int NIST_COMPOUND_FERROBORIDE = 59;
  public static final int NIST_COMPOUND_FERROUS_OXIDE = 60;
  public static final int NIST_COMPOUND_FERROUS_SULFATE_DOSIMETER_SOLUTION = 61;
  public static final int NIST_COMPOUND_FREON_12 = 62;
  public static final int NIST_COMPOUND_FREON_12B2 = 63;
  public static final int NIST_COMPOUND_FREON_13 = 64;
  public static final int NIST_COMPOUND_FREON_13B1 = 65;
  public static final int NIST_COMPOUND_FREON_13I1 = 66;
  public static final int NIST_COMPOUND_GADOLINIUM_OXYSULFIDE = 67;
  public static final int NIST_COMPOUND_GALLIUM_ARSENIDE = 68;
  public static final int NIST_COMPOUND_GEL_IN_PHOTOGRAPHIC_EMULSION = 69;
  public static final int NIST_COMPOUND_GLASS_PYREX = 70;
  public static final int NIST_COMPOUND_GLASS_LEAD = 71;
  public static final int NIST_COMPOUND_GLASS_PLATE = 72;
  public static final int NIST_COMPOUND_GLUCOSE = 73;
  public static final int NIST_COMPOUND_GLUTAMINE = 74;
  public static final int NIST_COMPOUND_GLYCEROL = 75;
  public static final int NIST_COMPOUND_GUANINE = 76;
  public static final int NIST_COMPOUND_GYPSUM_PLASTER_OF_PARIS = 77;
  public static final int NIST_COMPOUND_N_HEPTANE = 78;
  public static final int NIST_COMPOUND_N_HEXANE = 79;
  public static final int NIST_COMPOUND_KAPTON_POLYIMIDE_FILM = 80;
  public static final int NIST_COMPOUND_LANTHANUM_OXYBROMIDE = 81;
  public static final int NIST_COMPOUND_LANTHANUM_OXYSULFIDE = 82;
  public static final int NIST_COMPOUND_LEAD_OXIDE = 83;
  public static final int NIST_COMPOUND_LITHIUM_AMIDE = 84;
  public static final int NIST_COMPOUND_LITHIUM_CARBONATE = 85;
  public static final int NIST_COMPOUND_LITHIUM_FLUORIDE = 86;
  public static final int NIST_COMPOUND_LITHIUM_HYDRIDE = 87;
  public static final int NIST_COMPOUND_LITHIUM_IODIDE = 88;
  public static final int NIST_COMPOUND_LITHIUM_OXIDE = 89;
  public static final int NIST_COMPOUND_LITHIUM_TETRABORATE = 90;
  public static final int NIST_COMPOUND_LUNG_ICRP = 91;
  public static final int NIST_COMPOUND_M3_WAX = 92;
  public static final int NIST_COMPOUND_MAGNESIUM_CARBONATE = 93;
  public static final int NIST_COMPOUND_MAGNESIUM_FLUORIDE = 94;
  public static final int NIST_COMPOUND_MAGNESIUM_OXIDE = 95;
  public static final int NIST_COMPOUND_MAGNESIUM_TETRABORATE = 96;
  public static final int NIST_COMPOUND_MERCURIC_IODIDE = 97;
  public static final int NIST_COMPOUND_METHANE = 98;
  public static final int NIST_COMPOUND_METHANOL = 99;
  public static final int NIST_COMPOUND_MIX_D_WAX = 100;
  public static final int NIST_COMPOUND_MS20_TISSUE_SUBSTITUTE = 101;
  public static final int NIST_COMPOUND_MUSCLE_SKELETAL = 102;
  public static final int NIST_COMPOUND_MUSCLE_STRIATED = 103;
  public static final int NIST_COMPOUND_MUSCLE_EQUIVALENT_LIQUID_WITH_SUCROSE = 104;
  public static final int NIST_COMPOUND_MUSCLE_EQUIVALENT_LIQUID_WITHOUT_SUCROSE = 105;
  public static final int NIST_COMPOUND_NAPHTHALENE = 106;
  public static final int NIST_COMPOUND_NITROBENZENE = 107;
  public static final int NIST_COMPOUND_NITROUS_OXIDE = 108;
  public static final int NIST_COMPOUND_NYLON_DU_PONT_ELVAMIDE_8062 = 109;
  public static final int NIST_COMPOUND_NYLON_TYPE_6_AND_TYPE_66 = 110;
  public static final int NIST_COMPOUND_NYLON_TYPE_610 = 111;
  public static final int NIST_COMPOUND_NYLON_TYPE_11_RILSAN = 112;
  public static final int NIST_COMPOUND_OCTANE_LIQUID = 113;
  public static final int NIST_COMPOUND_PARAFFIN_WAX = 114;
  public static final int NIST_COMPOUND_N_PENTANE = 115;
  public static final int NIST_COMPOUND_PHOTOGRAPHIC_EMULSION = 116;
  public static final int NIST_COMPOUND_PLASTIC_SCINTILLATOR_VINYLTOLUENE_BASED = 117;
  public static final int NIST_COMPOUND_PLUTONIUM_DIOXIDE = 118;
  public static final int NIST_COMPOUND_POLYACRYLONITRILE = 119;
  public static final int NIST_COMPOUND_POLYCARBONATE_MAKROLON_LEXAN = 120;
  public static final int NIST_COMPOUND_POLYCHLOROSTYRENE = 121;
  public static final int NIST_COMPOUND_POLYETHYLENE = 122;
  public static final int NIST_COMPOUND_POLYETHYLENE_TEREPHTHALATE_MYLAR = 123;
  public static final int NIST_COMPOUND_POLYMETHYL_METHACRALATE_LUCITE_PERSPEX = 124;
  public static final int NIST_COMPOUND_POLYOXYMETHYLENE = 125;
  public static final int NIST_COMPOUND_POLYPROPYLENE = 126;
  public static final int NIST_COMPOUND_POLYSTYRENE = 127;
  public static final int NIST_COMPOUND_POLYTETRAFLUOROETHYLENE_TEFLON = 128;
  public static final int NIST_COMPOUND_POLYTRIFLUOROCHLOROETHYLENE = 129;
  public static final int NIST_COMPOUND_POLYVINYL_ACETATE = 130;
  public static final int NIST_COMPOUND_POLYVINYL_ALCOHOL = 131;
  public static final int NIST_COMPOUND_POLYVINYL_BUTYRAL = 132;
  public static final int NIST_COMPOUND_POLYVINYL_CHLORIDE = 133;
  public static final int NIST_COMPOUND_POLYVINYLIDENE_CHLORIDE_SARAN = 134;
  public static final int NIST_COMPOUND_POLYVINYLIDENE_FLUORIDE = 135;
  public static final int NIST_COMPOUND_POLYVINYL_PYRROLIDONE = 136;
  public static final int NIST_COMPOUND_POTASSIUM_IODIDE = 137;
  public static final int NIST_COMPOUND_POTASSIUM_OXIDE = 138;
  public static final int NIST_COMPOUND_PROPANE = 139;
  public static final int NIST_COMPOUND_PROPANE_LIQUID = 140;
  public static final int NIST_COMPOUND_N_PROPYL_ALCOHOL = 141;
  public static final int NIST_COMPOUND_PYRIDINE = 142;
  public static final int NIST_COMPOUND_RUBBER_BUTYL = 143;
  public static final int NIST_COMPOUND_RUBBER_NATURAL = 144;
  public static final int NIST_COMPOUND_RUBBER_NEOPRENE = 145;
  public static final int NIST_COMPOUND_SILICON_DIOXIDE = 146;
  public static final int NIST_COMPOUND_SILVER_BROMIDE = 147;
  public static final int NIST_COMPOUND_SILVER_CHLORIDE = 148;
  public static final int NIST_COMPOUND_SILVER_HALIDES_IN_PHOTOGRAPHIC_EMULSION = 149;
  public static final int NIST_COMPOUND_SILVER_IODIDE = 150;
  public static final int NIST_COMPOUND_SKIN_ICRP = 151;
  public static final int NIST_COMPOUND_SODIUM_CARBONATE = 152;
  public static final int NIST_COMPOUND_SODIUM_IODIDE = 153;
  public static final int NIST_COMPOUND_SODIUM_MONOXIDE = 154;
  public static final int NIST_COMPOUND_SODIUM_NITRATE = 155;
  public static final int NIST_COMPOUND_STILBENE = 156;
  public static final int NIST_COMPOUND_SUCROSE = 157;
  public static final int NIST_COMPOUND_TERPHENYL = 158;
  public static final int NIST_COMPOUND_TESTES_ICRP = 159;
  public static final int NIST_COMPOUND_TETRACHLOROETHYLENE = 160;
  public static final int NIST_COMPOUND_THALLIUM_CHLORIDE = 161;
  public static final int NIST_COMPOUND_TISSUE_SOFT_ICRP = 162;
  public static final int NIST_COMPOUND_TISSUE_SOFT_ICRU_FOUR_COMPONENT = 163;
  public static final int NIST_COMPOUND_TISSUE_EQUIVALENT_GAS_METHANE_BASED = 164;
  public static final int NIST_COMPOUND_TISSUE_EQUIVALENT_GAS_PROPANE_BASED = 165;
  public static final int NIST_COMPOUND_TITANIUM_DIOXIDE = 166;
  public static final int NIST_COMPOUND_TOLUENE = 167;
  public static final int NIST_COMPOUND_TRICHLOROETHYLENE = 168;
  public static final int NIST_COMPOUND_TRIETHYL_PHOSPHATE = 169;
  public static final int NIST_COMPOUND_TUNGSTEN_HEXAFLUORIDE = 170;
  public static final int NIST_COMPOUND_URANIUM_DICARBIDE = 171;
  public static final int NIST_COMPOUND_URANIUM_MONOCARBIDE = 172;
  public static final int NIST_COMPOUND_URANIUM_OXIDE = 173;
  public static final int NIST_COMPOUND_UREA = 174;
  public static final int NIST_COMPOUND_VALINE = 175;
  public static final int NIST_COMPOUND_VITON_FLUOROELASTOMER = 176;
  public static final int NIST_COMPOUND_WATER_LIQUID = 177;
  public static final int NIST_COMPOUND_WATER_VAPOR = 178;
  public static final int NIST_COMPOUND_XYLENE = 179;

  private static final String[] MendelArray = { "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh"
  };

  public static final int RADIO_NUCLIDE_55FE = 0;
  public static final int RADIO_NUCLIDE_57CO = 1;
  public static final int RADIO_NUCLIDE_109CD = 2;
  public static final int RADIO_NUCLIDE_125I = 3;
  public static final int RADIO_NUCLIDE_137CS = 4;
  public static final int RADIO_NUCLIDE_133BA = 5;
  public static final int RADIO_NUCLIDE_153GD = 6;
  public static final int RADIO_NUCLIDE_238PU = 7;
  public static final int RADIO_NUCLIDE_241AM = 8;
  public static final int RADIO_NUCLIDE_244CM = 9;

  private static final int[] LB_LINE_MACROS = new int[]{
    L2M4_LINE,
    L2M3_LINE,
    L3N5_LINE,
    L3O4_LINE,
    L3O5_LINE,
    L3O45_LINE,
    L3N1_LINE,
    L3O1_LINE,
    L3N6_LINE,
    L3N7_LINE,
    L3N4_LINE,
    L1M3_LINE,
    L1M2_LINE,
    L1M5_LINE,
    L1M4_LINE
  };

  private static class LineShellPair {
    public final int line;
    public final int shell;

    public LineShellPair(int line, int shell) {
      this.line = line;
      this.shell = shell;
    }
  }

  private static final LineShellPair[] lb_pairs = new LineShellPair[]{
    new LineShellPair(LB1_LINE, L2_SHELL),
    new LineShellPair(LB2_LINE, L3_SHELL),
    new LineShellPair(LB3_LINE, L1_SHELL),
    new LineShellPair(LB4_LINE, L1_SHELL),
    new LineShellPair(LB5_LINE, L3_SHELL),
    new LineShellPair(LB6_LINE, L3_SHELL),
    new LineShellPair(LB7_LINE, L3_SHELL),
    new LineShellPair(LB9_LINE, L1_SHELL),
    new LineShellPair(LB10_LINE, L1_SHELL),
    new LineShellPair(LB15_LINE, L3_SHELL),
    new LineShellPair(LB17_LINE, L2_SHELL),
    new LineShellPair(L3N6_LINE, L3_SHELL),
    new LineShellPair(L3N7_LINE, L3_SHELL),
  };

  private static final int KL1 = -KL1_LINE - 1;
  private static final int KL3 = -KL3_LINE - 1;
  private static final int KM1 = -KM1_LINE - 1;
  private static final int KP5 = -KP5_LINE - 1;

  private Xraylib() {
    // intentionally empty constructor
  }

}
