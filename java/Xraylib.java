/*
Copyright (c) 2015, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.BufferUnderflowException;
import java.nio.ByteOrder;
import java.lang.Math;

public class Xraylib {

  private static double[][] readDoubleArrayOfArrays(int[] N, ByteBuffer byte_buffer) throws BufferUnderflowException {
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

  private static double[] readDoubleArray(int n, ByteBuffer byte_buffer) throws BufferUnderflowException {
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

  private static int[] readIntArray(int n, ByteBuffer byte_buffer) throws BufferUnderflowException {
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



  public static void XRayInit() {
    try {
      DataInputStream inputStream = new DataInputStream(Xraylib.class.getClassLoader().getResourceAsStream("xraylib.dat"));
      int bytes_total = inputStream.available();
      byte[] bytes = new byte[bytes_total];
      inputStream.readFully(bytes);
      ByteBuffer byte_buffer = ByteBuffer.wrap(bytes);
      byte_buffer.order(ByteOrder.LITTLE_ENDIAN);
      ZMAX = byte_buffer.getInt();
      SHELLNUM = byte_buffer.getInt();
      TRANSNUM = byte_buffer.getInt();
      LINENUM = byte_buffer.getInt();
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

      //this should never happen!
      if (byte_buffer.hasRemaining()) {
        throw new RuntimeException("byte_buffer not empty when closing!");
      }

      inputStream.close();
	  }
    catch (IOException | RuntimeException e ) {
      e.printStackTrace();
    }
  }

  public static double AtomicWeight(int Z) {
    double atomic_weight;

    if (Z < 1 || Z > ZMAX) {
      throw new XraylibException("Z out of range");
    }

    atomic_weight = AtomicWeight_arr[Z];

    if (atomic_weight < 0.) {
      throw new XraylibException("Atomic Weight not available");
    }

    return atomic_weight;
  }

  public static double ElementDensity(int Z) {
    double element_density;

    if (Z<1 || Z>ZMAX) {
      throw new XraylibException("Z out of range");
    }

    element_density = ElementDensity_arr[Z];

    if (element_density < 0.) {
      throw new XraylibException("Element density not available");
    }

    return element_density;
  }

  public static double EdgeEnergy(int Z, int shell) {
    double edge_energy;

    if (Z<1 || Z>ZMAX) {
      throw new XraylibException("Z out of range");
    }

    if (shell<0 || shell>=SHELLNUM) {
      throw new XraylibException("Shell not available");
    }

    edge_energy = EdgeEnergy_arr[shell + (Z*SHELLNUM)];

    if (edge_energy < 0.) {
      throw new XraylibException("Edge energy not available");
    }

    return edge_energy;
  }

  public static double AtomicLevelWidth(int Z, int shell) {
    double atomic_level_width;

    if (Z<1 || Z>ZMAX) {
      throw new XraylibException("Z out of range");
    }

    if (shell<0 || shell>=SHELLNUM) {
      throw new XraylibException("Shell not available");
    }

    atomic_level_width = AtomicLevelWidth_arr[Z*SHELLNUM + shell];

    if (atomic_level_width < 0.) {
      throw new XraylibException("Shell not available");
    }

    return atomic_level_width;
  }

  private static double CS_Factory(int Z, double E, int[] NE_arr, double[][] E_arr, double[][] CS_arr, double[][] CS_arr2) throws XraylibException {
    double ln_E, ln_sigma, sigma;

    if (Z < 1 || Z > ZMAX || NE_arr[Z] < 0) {
      throw new XraylibException("Z out of range");
    }

    if (E <= 0.) {
      throw new XraylibException("Energy <=0");
    }

    ln_E = Math.log(E * 1000.0);

    ln_sigma = splint(E_arr[Z], CS_arr[Z], CS_arr2[Z], NE_arr[Z], ln_E);

    sigma = Math.exp(ln_sigma);

    return sigma;
  }

  public static double CS_Photo(int Z, double E) {
    try {
      double rv = CS_Factory(Z, E, NE_Photo_arr, E_Photo_arr, CS_Photo_arr, CS_Photo_arr2);
      return rv;
    }
    catch (XraylibException e) {
      throw new XraylibException(e.getMessage());
    }
  }

  public static double CS_Rayl(int Z, double E) {
    try {
      double rv = CS_Factory(Z, E, NE_Rayl_arr, E_Rayl_arr, CS_Rayl_arr, CS_Rayl_arr2);
      return rv;
    }
    catch (XraylibException e) {
      throw new XraylibException(e.getMessage());
    }
  }

  public static double CS_Compt(int Z, double E) {
    try {
      double rv = CS_Factory(Z, E, NE_Compt_arr, E_Compt_arr, CS_Compt_arr, CS_Compt_arr2);
      return rv;
    }
    catch (XraylibException e) {
      throw new XraylibException(e.getMessage());
    }
  }

  public static double CS_Energy(int Z, double E) {
    try {
      double rv = CS_Factory(Z, E/1000.0, NE_Energy_arr, E_Energy_arr, CS_Energy_arr, CS_Energy_arr2);
      return rv;
    }
    catch (XraylibException e) {
      throw new XraylibException(e.getMessage());
    }
  }

  public static double CS_Total(int Z, double E) {
    if (Z<1 || Z>ZMAX || NE_Photo_arr[Z]<0 || NE_Rayl_arr[Z]<0 || NE_Compt_arr[Z]<0) {
      throw new XraylibException("Z out of range");
    }

    if (E <= 0.) {
      throw new XraylibException("Energy <=0");
    }

    return CS_Photo(Z, E) + CS_Rayl(Z, E) + CS_Compt(Z, E);
  }

  private static double splint(double[] xa, double[] ya, double[] y2a, int n, double x) {
    int klo, khi, k;
    double h, b, a;

    if (x >= xa[n-1]) {
	    return ya[n-1];
    }

    if (x <= xa[0]) {
      return ya[0];
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

  public static int ZMAX;
  public static int SHELLNUM;
  public static int TRANSNUM;
  public static int LINENUM;

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
}
