#!/usr/bin/env python3


import urllib.request
from io import BytesIO
import lxml.etree as ET

nist_url = "http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=";

with open('../densities.dat', 'w') as output:
    for i in range(1, 99):
        url = '{}{:03}.html'.format(nist_url, i)
        print("URL: {}".format(url))
        try:
            response = urllib.request.urlopen(url)
            html = response.read()
            #print(html)
            parser = ET.HTMLParser()
            tree = ET.parse(BytesIO(html), parser)
            density = tree.xpath("//body/center/table/tr[1]/td[2]/text()")
            output.write("{}\t{:f}\n".format(i, float(density[0])))
        except Exception as e:
            print(e)
            raise




