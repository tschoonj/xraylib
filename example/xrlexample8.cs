using System;
using System.Diagnostics;
using Science;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            XrayLib xl = XrayLib.Instance;
            // If something goes wrong, the test will end with EXIT_FAILURE
//            xl.SetHardExit(1);
            
            Console.Title = String.Format("XrayLib.NET v{0}.{1}", 
                XrayLib.VERSION_MAJOR, XrayLib.VERSION_MINOR);
            Console.WriteLine("Example C# program using XrayLib.NET\n");
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

            Console.WriteLine("Compton profile for Fe at pz = 1.1 : {0}", 
                xl.ComptonProfile(26, 1.1f));
            Console.WriteLine("M5 Compton profile for Fe at pz = 1.1 : {0}", 
                xl.ComptonProfile_Partial(26, XrayLib.M5_SHELL, 1.1f));

            sw.Stop();
            Console.WriteLine("Time: {0} ms", sw.ElapsedMilliseconds);

            Console.ReadLine();
        }
    }
}

/*
  
  

  //parser test for SiO2 (quartz)
  if (CompoundParser("SiO2",&cdtest) == 0)
	return 1;

  std::printf("SiO2 contains %i atoms and %i elements\n",cdtest.nAtomsAll,cdtest.nElements);
  for (i = 0 ; i < cdtest.nElements ; i++)
    std::printf("Element %i: %lf %%\n",cdtest.Elements[i],cdtest.massFractions[i]*100.0);

  FREE_COMPOUND_DATA(cdtest)

  std::printf("Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f\n",CS_Rayl_CP("Ca(HCO3)2",10.0f) );

  std::printf("CS2 Refractive Index at 10.0 keV : %f - %f i\n",Refractive_Index_Re("CS2",10.0f,1.261f),Refractive_Index_Im("CS2",10.0f,1.261f));
  std::printf("C16H14O3 Refractive Index at 1 keV : %f - %f i\n",Refractive_Index_Re("C16H14O3",1.0f,1.2f),Refractive_Index_Im("C16H14O3",1.0f,1.2f));
  std::printf("SiO2 Refractive Index at 5 keV : %f - %f i\n",Refractive_Index_Re("SiO2",5.0f,2.65f),Refractive_Index_Im("SiO2",5.0f,2.65f));
*/

