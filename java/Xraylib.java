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

  public static double FluorYield(int Z, int shell) {
    double fluor_yield;

    if (Z<1 || Z>ZMAX) {
      throw new XraylibException("Z out of range");
    }

    if (shell<0 || shell>=SHELLNUM) {
      throw new XraylibException("Shell not available");
    }

    fluor_yield = FluorYield_arr[Z*SHELLNUM + shell];

    if (fluor_yield < 0.) {
      throw new XraylibException("Shell not available");
    }

    return fluor_yield;
  }

  public static double JumpFactor(int Z, int shell) {
    double jump_factor;

    if (Z<1 || Z>ZMAX) {
      throw new XraylibException("Z out of range");
    }

    if (shell<0 || shell>=SHELLNUM) {
      throw new XraylibException("Shell not available");
    }

    jump_factor = JumpFactor_arr[Z*SHELLNUM + shell];

    if (jump_factor < 0.) {
      throw new XraylibException("Shell not available");
    }

    return jump_factor;
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
}
