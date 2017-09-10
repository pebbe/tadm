# $Id: Makefile,v 1.5 2007/09/11 21:10:19 jasonbaldridge Exp $

ALL: tadm evaluate

all: ALL

OBJS = tadm.o dataset.o probs.o model.o mle.o fileio.o iis.o steep.o perceptron.o fix.o

LDFLAGS = $(shell pkg-config --libs-only-L --libs-only-other PETSc)
CXXFLAGS = $(shell pkg-config --cflags PETSc)
LDLIBS = $(shell pkg-config --libs-only-l PETSc) -lz

CC = g++

fileio.o: fileio.cc fileio.h fix.h
dataset.o: dataset.cc fileio.h tadm.h fix.h
tadm.o: tadm.cc tadm.h version.h fix.h
probs.o: probs.cc tadm.h fix.h
model.o: model.cc tadm.h fix.h
mle.o: mle.cc tadm.h fix.h
iis.o: iis.cc tadm.h fix.h
perceptron.o: perceptron.cc tadm.h fix.h
steep.o: steep.cc fix.h
svd.o: svd.cc svd.h las2.h fix.h
las2.o: las2.cc las2.h fix.h

tadm: $(OBJS)

rosenbrock: rosenbrock.o steep.o 

evaluate: fileio.o evaluate.o
