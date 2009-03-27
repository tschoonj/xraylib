#!/bin/sh

rm -rf /usr/local/xraylib
rm -rf /usr/local/include/xraylib.h /usr/local/include/xraylib.mod /usr/local/lib64/libxrl.so  /usr/local/include/shells.h /usr/local/include/lines.h
cp -r /home/schoon/.xraylib_v2.11 /usr/local/xraylib

ln -s /usr/local/xraylib/include/xraylib.h /usr/local/include/xraylib.h
ln -s /usr/local/xraylib/include/lines.h /usr/local/include/lines.h
ln -s /usr/local/xraylib/include/shells.h /usr/local/include/shells.h
ln -s /usr/local/xraylib/include/xraylib.mod /usr/local/include/xraylib.mod
#should be arch dependent
ln -s /usr/local/xraylib/lib/libxrl.so /usr/local/lib64/libxrl.so
