#-------------------------------------------------
#
# Project created by QtCreator 2014-03-06T11:33:19
#
#-------------------------------------------------

QT       += core gui\
	network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = 'Feature-based image classificator'
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    codebookworker.cpp \
    trainworker.cpp \
    traincontroller.cpp \
    codebookcontroller.cpp \
    logstream.cpp \
    utils.cpp

HEADERS  += mainwindow.h \
    codebookworker.h \
    trainworker.h \
    traincontroller.h \
    codebookcontroller.h \
    logstream.h \
    utils.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -fopenmp -L/usr/local/lib/vl
QMAKE_LFLAGS +=  -fopenmp -L/usr/local/lib/vl
QMAKE_CFLAGS_DEBUG   += -fopenmp -L/usr/local/lib/vl
QMAKE_CFLAGS_RELEASE +=  -fopenmp -L/usr/local/lib/vl

install_it.path = $$OUT_PWD
install_it.files = *.wav .*.wav *.ico

INSTALLS += \
    install_it

QMAKE_POST_LINK = cp $$PWD/*.wav $$PWD/.*.wav $$PWD/*.ico $$OUT_PWD/

LIBS += `pkg-config opencv --libs` -lvl
