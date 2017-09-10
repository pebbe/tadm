# $Id: Makefile,v 1.5 2007/09/11 21:10:19 jasonbaldridge Exp $

ALL: tadm evaluate

include ${TAO_DIR}/bmake/tao_common

OBJS = tadm.o dataset.o probs.o model.o mle.o fileio.o iis.o steep.o perceptron.o

fileio.o: fileio.cc fileio.h
dataset.o: dataset.cc fileio.h tadm.h
tadm.o: tadm.cc tadm.h version.h
probs.o: probs.cc tadm.h
model.o: model.cc tadm.h
mle.o: mle.cc tadm.h
iis.o: iis.cc tadm.h
perceptron.o: perceptron.cc tadm.h
steep.o: steep.cc
svd.o: svd.cc svd.h las2.h
las2.o: las2.cc las2.h

tadm: $(OBJS)
	-${CLINKER} ${LDFLAGS} -o ../bin/tadm $(OBJS) ${TAO_LIB} ${PETSC_LIB} -lz

rosenbrock: rosenbrock.o steep.o 
	-${CLINKER} ${LDFLAGS} -o rosenbrock rosenbrock.o steep.o ${TAO_LIB} ${PETSC_LIB}

evaluate: fileio.o evaluate.o
	-${CLINKER} ${LDFLAGS} -o ../bin/evaluate fileio.o evaluate.o -lz
