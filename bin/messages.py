def display_banner():
    print
    fp = open('../doc/xraybanner.txt', 'r')
    for line in fp.readlines():
        print line,
    fp.close()
    print
def display_options():
    print
    print " - Type 'xraylib -h' to see a list of the available functions"
    print " - Type 'xraylib -d' to see the X-ray data documentation"
    print " - Type 'xraylib -f function-name' to get help on a" \
          " specific function" 

def display_usage():
    print
    print " usage: xraylib 'expression'"
    print "     where 'expression' is any mathematical expression"
    print "     that can contain X-ray library functions."
    display_options()

def display_help():
    print
    print "Available X-ray library functions"
    print
    fp = open('../doc/xrayfunc.txt', 'r')
    for line in fp.readlines():
        print line,
    fp.close()
    display_usage()

def display_doc():
    print
    print "X-ray data documentation"
    print
    fp = open('../doc/xraydoc.txt', 'r')
    for line in fp.readlines():
        print line,
    fp.close()
    print

