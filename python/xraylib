from math import *
from _xraylib import *
import getopt, sys, string, traceback
from messages import *
from xrayhelp import *

if __name__ == '__main__' :
    short = 'hdf:'
    long = ('help', 'doc', 'func=')
    arglist = string.split(sys.argv[1])
    opts,args = getopt.getopt(arglist, short, long)
    for opt,val in opts:
        if opt in ('-h', '--help'):
            display_banner()
	    display_help()
	    sys.exit(0)
        elif opt in ('-d', '--doc'):
            display_banner()
	    display_doc()
	    sys.exit(0)
        elif opt in ('-f', '--func'):
            print val
	    display_func(val)
	    sys.exit(0)
    if sys.argv[1] == '':
        display_banner()
        display_usage()
        sys.exit(0)
    try:    
        XRayInit()
        print "%.5g" % eval(sys.argv[1])
    except:
        traceback.print_exc()
        display_usage()
