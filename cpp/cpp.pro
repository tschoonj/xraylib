TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

HEADERS += \
    xraylib_test.h

# include xraylib
INCLUDEPATH += /usr/local/xraylib/include/xraylib
LIBS += -L/usr/local/xraylib/lib/ -lxrl

# include googletest
INCLUDEPATH += /usr/local/include/
LIBS += -L/usr/local/lib/ -lgtest

DISTFILES += \
    generate-code.py
