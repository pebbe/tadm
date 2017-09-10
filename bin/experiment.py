#!/usr/bin/python

#################################################################################
## The Toolkit for Advanced Discriminative Modeling
## Copyright (C) 2001-2005 Robert Malouf
## 
## This library is free software; you can redistribute it and#or
## modify it under the terms of the GNU Lesser General Public
## License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## 
## This library is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public
## License along with this library; if not, write to the Free Software
## Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#################################################################################

"""Example of using TADM to build MaxEnt classifiers."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "13 October 2002"
__version__ = "$Revision: 1.1.1.1 $"

import psyco
psyco.full()
from psyco.classes import *

import sys
import os
import getopt
from maxent import *


train = RainbowSource('text.train',binary=True)
test = RainbowSource('text.test',binary=True)

log = open('experiment.out','w')

var = 0.001
for i in xrange(1,10,1):
    var = var * 10.0
    me = MaxEntInducer().train(train,var=var)
    acc = AccuracySink()
    me.apply(test,acc)
    print '****'
    print var, acc.score()*100.0
    print >> log, var, acc.score()*100.0
