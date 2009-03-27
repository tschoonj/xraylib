pro xrlh
spawn, 'more $XRAYLIB_DIR/doc/xrayfunc.txt'
print, " - Type xrlh to see a list of the available functions"
print, " - Type xrld to see the X-ray data documentation"
print, " - Type xrlf, 'function-name' to get help on a specific function"
end

pro xrld
spawn, 'xraylib -d'
end

pro xrlf, funcname
command = 'xraylib -f ' + funcname
spawn, command
end

