using System;
using System.Diagnostics;
using System.Numerics;
using Science;
using System.Collections.Generic;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            XrayLib xl = XrayLib.Instance;
           
            Console.Title = String.Format("XrayLib.NET v{0}.{1}", 
                XrayLib.VERSION_MAJOR, XrayLib.VERSION_MINOR);
            Console.WriteLine("Example C# program using XrayLib.NET\n");
            Console.WriteLine("Density of pure Al: {0} g/cm3", 
                xl.ElementDensity(13));
            Console.WriteLine("Ca K-alpha Fluorescence Line Energy: {0}", 
                xl.LineEnergy(20, XrayLib.KA_LINE));
            Console.WriteLine("Fe partial photoionization cs of L3 at 6.0 keV: {0}", 
                xl.CS_Photo_Partial(26, XrayLib.L3_SHELL, 6.0));
            Console.WriteLine("Zr L1 edge energy: {0}", 
                xl.EdgeEnergy(40, XrayLib.L1_SHELL));
            Console.WriteLine("Pb Lalpha XRF production cs at 20.0 keV (jump approx): {0}", 
                xl.CS_FluorLine(82, XrayLib.LA_LINE, 20.0));
            Console.WriteLine("Pb Lalpha XRF production cs at 20.0 keV (Kissel): {0}",
                xl.CS_FluorLine_Kissel(82, XrayLib.LA_LINE, 20.0));
            Console.WriteLine("Bi M1N2 radiative rate: {0}", 
                xl.RadRate(83, XrayLib.M1N2_LINE));
            Console.WriteLine("U M3O3 Fluorescence Line Energy: {0}",
                xl.LineEnergy(92, XrayLib.M3O3_LINE));
            
            Console.WriteLine("Pb information: {0}", 
                xl.GetElementData(82).ToString());

            // Parser test for Ca(HCO3)2 (calcium bicarbonate)
            CompoundData cd = new CompoundData("Ca(HCO3)2");
            Console.WriteLine("Ca(HCO3)2 contains:");
            Console.Write(cd.ToString());

            // Parser test for SiO2 (quartz)
            cd.Parse("SiO2");
            Console.WriteLine("SiO2 contains:");
            Console.Write(cd.ToString());

            Console.WriteLine("Ca(HCO3)2 Rayleigh cs at 10.0 keV: {0}",
                xl.CS_Rayl_CP("Ca(HCO3)2", 10.0));
            Console.WriteLine("CS2 Refractive Index at 10.0 keV : {0} - {1} i",
                xl.Refractive_Index_Re("CS2", 10.0, 1.261), xl.Refractive_Index_Im("CS2", 10.0, 1.261));
            Console.WriteLine("C16H14O3 Refractive Index at 1 keV : {0} - {1} i",
                xl.Refractive_Index_Re("C16H14O3", 1.0, 1.2), xl.Refractive_Index_Im("C16H14O3", 1.0, 1.2));
            Console.WriteLine("SiO2 Refractive Index at 5 keV : {0} - {1} i",
                xl.Refractive_Index_Re("SiO2", 5.0, 2.65), xl.Refractive_Index_Im("SiO2", 5.0, 2.65));

            Complex n = xl.Refractive_Index("SiO2", 5.0, 2.65);
            Console.WriteLine("SiO2 Refractive Index at 5 keV : {0} - {1} i", n.Real, n.Imaginary);

            Console.WriteLine("Compton profile for Fe at pz = 1.1 : {0}", 
                xl.ComptonProfile(26, 1.1f));
            Console.WriteLine("M5 Compton profile for Fe at pz = 1.1 : {0}", 
                xl.ComptonProfile_Partial(26, XrayLib.M5_SHELL, 1.1));
            Console.WriteLine("M1->M5 Coster-Kronig transition probability for Au : {0}",
                xl.CosKronTransProb(79, XrayLib.FM15_TRANS));
            Console.WriteLine("L1->L3 Coster-Kronig transition probability for Fe : {0}",
                xl.CosKronTransProb(26, XrayLib.FL13_TRANS));
            Console.WriteLine("Au Ma1 XRF production cs at 10.0 keV (Kissel): {0}", 
                xl.CS_FluorLine_Kissel(79, XrayLib.MA1_LINE, 10.0));
            Console.WriteLine("Au Mb XRF production cs at 10.0 keV (Kissel): {0}", 
                xl.CS_FluorLine_Kissel(79, XrayLib.MB_LINE, 10.0));
            Console.WriteLine("Au Mg XRF production cs at 10.0 keV (Kissel): {0}", 
                xl.CS_FluorLine_Kissel(79, XrayLib.MG_LINE, 10.0));

            Console.WriteLine("K atomic level width for Fe: {0}", 
                xl.AtomicLevelWidth(26, XrayLib.K_SHELL));
            Console.WriteLine("Bi L2-M5M5 Auger non-radiative rate: {0}",
                xl.AugerRate(86, XrayLib.L2_M5M5_AUGER));

            Console.WriteLine("Sr anomalous scattering factor Fi at 10.0 keV: {0}", xl.Fi(38, 10.0));
            Console.WriteLine("Sr anomalous scattering factor Fii at 10.0 keV: {0}", xl.Fii(38, 10.0));

            cd = new CompoundData("SiO2", 0.4, "Ca(HCO3)2", 0.6);
            Console.WriteLine("Compound contains:");
            Console.Write(cd.ToString());

            String symbol = CompoundData.AtomicNumberToSymbol(26);
            Console.WriteLine("Symbol of element 26 is: {0}", symbol);
            Console.WriteLine("Number of element Fe is: {0}", CompoundData.SymbolToAtomicNumber("Fe"));

            Console.WriteLine("Pb Malpha XRF production cs at 20.0 keV with cascade effect: {0}",
                xl.CS_FluorLine_Kissel(82, XrayLib.MA1_LINE, 20.0));
            Console.WriteLine("Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: {0}", 
                xl.CS_FluorLine_Kissel_Radiative_Cascade(82, XrayLib.MA1_LINE, 20.0));
            Console.WriteLine("Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: {0}", 
                xl.CS_FluorLine_Kissel_Nonradiative_Cascade(82, XrayLib.MA1_LINE, 20.0));
            Console.WriteLine("Pb Malpha XRF production cs at 20.0 keV without cascade effect: {0}", 
                xl.CS_FluorLine_Kissel_No_Cascade(82, XrayLib.MA1_LINE, 20.0));

            Console.WriteLine("Al mass energy-absorption cs at 20.0 keV: {0}", 
                xl.CS_Energy(13, 20.0));
            Console.WriteLine("Pb mass energy-absorption cs at 40.0 keV: {0}", 
                xl.CS_Energy(82, 40.0));
            Console.WriteLine("CdTe mass energy-absorption cs at 40.0 keV: {0}", 
                xl.CS_Energy_CP("CdTe", 40.0));

            double energy = 8.0;
            double debyeFactor = 1.0;
            double relativeAngle = 1.0;
            
            // Si crystal structure
            CrystalArray ca = new CrystalArray();

            Crystal cryst = ca.GetCrystal("Si");
            if (cryst != null)
            {
                Console.WriteLine(cryst.ToString());

                // Si diffraction parameters
                Console.WriteLine("Si 111 at 8 KeV. Incidence at the Bragg angle:");
                double bragg = cryst.BraggAngle(energy, 1, 1, 1);
                Console.WriteLine("  Bragg angle: {0} rad, {1} deg", bragg, bragg*180.0/Math.PI);
                
                double q = cryst.ScatteringVectorMagnitide(energy, 1, 1, 1, relativeAngle);
                Console.WriteLine("  Magnitude of scattering vector, Q: {0}", q);

                double f0 = 0.0, fp = 0.0, fpp = 0.0;
                cryst.AtomicScatteringFactors(14, energy, q, debyeFactor, ref f0, ref fp, ref fpp);
                Console.WriteLine(" Atomic scattering factors (Z = 14) f0, fp, fpp: {0}, {1}, i{2}", f0, fp, fpp);

                Complex FH, F0;
                FH = cryst.StructureFactor(energy, 1, 1, 1, debyeFactor, relativeAngle);
                Console.WriteLine("  FH(1,1,1) structure factor: ({0}, {1})", FH.Real, FH.Imaginary);

                F0 = cryst.StructureFactor(energy, 0, 0, 0, debyeFactor, relativeAngle);
                Console.WriteLine("  F0=FH(0,0,0) structure factor: ({0}, {1})", F0.Real, F0.Imaginary);
                Console.WriteLine();
            }

            // Diamond diffraction parameters
            cryst = ca.GetCrystal("Diamond");
            if (cryst != null)
            {
                Console.WriteLine("Diamond 111 at 8 KeV. Incidence at the Bragg angle:"); 
                double bragg = cryst.BraggAngle(energy, 1, 1, 1);
                Console.WriteLine("  Bragg angle: {0} rad, {1} deg", bragg, bragg * 180.0 / Math.PI);
                
                double q = cryst.ScatteringVectorMagnitide(energy, 1, 1, 1, relativeAngle);
                Console.WriteLine("  Magnitude of scattering vector, Q: {0}", q);

                double f0 = 0.0, fp = 0.0, fpp = 0.0;
                cryst.AtomicScatteringFactors(6, energy, q, debyeFactor, ref f0, ref fp, ref fpp);
                Console.WriteLine(" Atomic scattering factors (Z = 6) f0, fp, fpp: {0}, {1}, i{2}", f0, fp, fpp);

                Complex FH, F0;
                FH = cryst.StructureFactor(energy, 1, 1, 1, debyeFactor, relativeAngle);
                Console.WriteLine("  FH(1,1,1) structure factor: ({0}, {1})", FH.Real, FH.Imaginary);

                F0 = cryst.StructureFactor(energy, 0, 0, 0, debyeFactor, relativeAngle);
                Console.WriteLine("  F0=FH(0,0,0) structure factor: ({0}, {1})", F0.Real, F0.Imaginary);
                

                Complex FHbar = cryst.StructureFactor(energy, -1, -1, -1, debyeFactor, relativeAngle);
                double dw = 1e10 * 2 * (XrayLib.R_E / cryst.Volume) * (XrayLib.KEV2ANGST * XrayLib.KEV2ANGST / (energy * energy)) *
                                                                Math.Sqrt(Complex.Abs(FH * FHbar)) / Math.PI / Math.Sin(2 * bragg);
                Console.WriteLine("  Darwin width: {0} uRad", 1e6 * dw); 
                Console.WriteLine();
            }

            // Alpha Quartz diffraction parameters
            cryst = ca.GetCrystal("AlphaQuartz");
            if (cryst != null)
            {
                Console.WriteLine("AlphaQuartz 020 at 8 KeV. Incidence at the Bragg angle:"); 
                double bragg = cryst.BraggAngle(energy, 0, 2, 0);
                Console.WriteLine("  Bragg angle: {0} rad, {1} deg", bragg, bragg * 180.0 / Math.PI);
                
                double q = cryst.ScatteringVectorMagnitide(energy, 0, 2, 0, relativeAngle);
                Console.WriteLine("  Magnitude of scattering vector, Q: {0}", q);

                double f0 = 0.0, fp = 0.0, fpp = 0.0;
                cryst.AtomicScatteringFactors(8, energy, q, debyeFactor, ref f0, ref fp, ref fpp);
                Console.WriteLine(" Atomic scattering factors (Z = 8) f0, fp, fpp: {0}, {1}, i{2}", f0, fp, fpp);

                Complex FH, F0;
                FH = cryst.StructureFactor(energy, 0, 2, 0, debyeFactor, relativeAngle);
                Console.WriteLine("  FH(0,2,0) structure factor: ({0}, {1})", FH.Real, FH.Imaginary);

                F0 = cryst.StructureFactor(energy, 0, 0, 0, debyeFactor, relativeAngle);
                Console.WriteLine("  F0=FH(0,0,0) structure factor: ({0}, {1})", F0.Real, F0.Imaginary);
                Console.WriteLine();
            }

            // Muscovite diffraction parameters
            cryst = ca.GetCrystal("Muscovite");
            if (cryst != null)
            {
                Console.WriteLine("Muskovite 331 at 8 KeV. Incidence at the Bragg angle:");
                double bragg = cryst.BraggAngle(energy, 3, 3, 1);
                Console.WriteLine("  Bragg angle: {0} rad, {1} deg", bragg, bragg * 180.0 / Math.PI);

                double q = cryst.ScatteringVectorMagnitide(energy, 3, 3, 1, relativeAngle);
                Console.WriteLine("  Magnitude of scattering vector, Q: {0}", q);

                double f0 = 0.0, fp = 0.0, fpp = 0.0;
                cryst.AtomicScatteringFactors(19, energy, q, debyeFactor, ref f0, ref fp, ref fpp);
                Console.WriteLine(" Atomic scattering factors (Z = 19) f0, fp, fpp: {0}, {1}, i{2}", f0, fp, fpp);

                Complex FH, F0;
                FH = cryst.StructureFactor(energy, 3, 3, 1, debyeFactor, relativeAngle);
                Console.WriteLine("  FH(3,3,1) structure factor: ({0}, {1})", FH.Real, FH.Imaginary);

                F0 = cryst.StructureFactor(energy, 0, 0, 0, debyeFactor, relativeAngle);
                Console.WriteLine("  F0=FH(0,0,0) structure factor: ({0}, {1})", F0.Real, F0.Imaginary);
                Console.WriteLine();
            }

            List<string> crystalNames;
            crystalNames = CrystalArray.GetDefaultNames();
            foreach (string name in crystalNames)
                Console.WriteLine(name);
            Console.WriteLine(); 

            // RadionuclideData tests 
            RadionuclideData rd = new RadionuclideData("109Cd");
            Console.WriteLine(rd.ToString());
            Console.WriteLine();

            rd = new RadionuclideData(XrayLib.RADIONUCLIDE_125I);
            Console.WriteLine(rd.ToString());
            Console.WriteLine();

            rd = new RadionuclideData();
            string namesCsv = string.Join(", ", rd.Names.ToArray());
            Console.WriteLine(namesCsv);
            Console.WriteLine();

            sw.Stop();
            Console.WriteLine("Time: {0} ms", sw.ElapsedMilliseconds);

            Console.ReadLine();
        }
    }
}

