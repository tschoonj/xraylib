#!/usr/bin/env python3

# taken from https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/

import os
import sys
import shutil

'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

build_root = os.environ['MESON_BUILD_ROOT']
install_destdir_prefix = os.environ['MESON_INSTALL_DESTDIR_PREFIX']
files = getListOfFiles(build_root)
files = list(filter(lambda file: os.path.basename(file) == 'xraylib.mod', files))
if not files:
    sys.exit('xraylib.mod was not found!')

shutil.copy(files[0], os.path.join(install_destdir_prefix, 'include', 'xraylib'))



