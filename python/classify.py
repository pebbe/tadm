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

"""Example of using TADM to build classifiers."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "12 February 2004"
__version__ = "$Revision: 1.7 $"

import cPickle
import optparse
import sys

from tadm.datasource import *
from tadm.inducer import *
from tadm.model import *
from tadm.result_sink import *

try:
    import psyco
    psyco.full()
    from psyco.classes import *
except:
    pass

def main():
    # process options

    parser = optparse.OptionParser()
    parser.add_option("-i", "--max-iteration", type="int", default=0,
                      help="perform a maximum number of iterations during training",
                      metavar="NUM")
    parser.add_option("-p", "--prior", type="float", default=0,
                      help="use Gaussion prior with variance VAR",
                      metavar="VAR")
    parser.add_option("-l", "--lasso", type="float", default=0,
                      help="use least absolute smoothing with variance VAR",
                      metavar="VAR")
    parser.add_option("-a", "--algorithm", choices=['MaxEnt','MaxEntRanker', 'NaiveBayes','Perceptron'],
                      default='MaxEnt',
                      help="use learning algoritm ALG (MaxEnt (def), MaxEntRanker, NaiveBayes, Perceptron)",
                      metavar="ALG")
    parser.add_option("-d", "--data-format", choices=['Simple','SimpleRanker','C45','Sparse','Rainbow'],
                      default='Simple',
                      help="use data format FORM (Simple (def), SimpleRanker, C45, Sparse, Rainbow)",
                      metavar="FORM")
    parser.add_option("-s", "--smooth", type="float", default=0,
                      help="add EPS to all counts",
                      metavar="EPS")
    parser.add_option("-m", "--model", action="store", default='',
                      help="use model FILE",
                      metavar="FILE")
    parser.add_option("-f", "--train", action="store", default='',
                      help="read training data from FILE",
                      metavar="FILE")
    parser.add_option("-t", "--test", action="store", default='',
                      help="read test data from FILE",
                      metavar="FILE")
    parser.add_option("-q", "--quiet", action="store_true", default=False,
                      help="be quiet")
    parser.add_option("-b", "--no-baseline", action="store_true", default=False,
                      help="don't use a baseline category")
    parser.add_option("-c", "--confusion", action="store_true", default=False,
                      help="print confusion matrix")
    parser.add_option("-y", "--display-parameters", action="store_true", default=False,
                      help="display all parameters")
    parser.add_option("-z", "--display-predictions", action="store_true", default=False,
                      help="display model predictions on the evaluation file")
    parser.add_option("-x", "--display-label-distributions", action="store_true", default=False,
                      help="display model's distribution over all labels on the evaluation file")
    (options, args) = parser.parse_args()

    # Construct model

    if (options.train == ''):
        # no training file, so check model file
        if (options.model != ''):
            if (not options.quiet):
                print >> sys.stderr, 'Reading model from',options.model
            file = open(options.model,'r')
            me = cPickle.load(file)
            file.close()
        else:
            raise Exception
    else:
        if (options.data_format == 'Simple'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Simple training data from', \
                      options.train
            source = SimpleSource(options.train)
        elif (options.data_format == 'SimpleRanker'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading SimpleRanker training data from', \
                      options.train
            source = SimpleRankerSource(options.train)
        elif (options.data_format == 'Rainbow'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Rainbow training data from', \
                      options.train
            source = RainbowSource(options.train,binary=True)
        elif (options.data_format == 'C45'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading C4.5 training data from', \
                      options.train
            source = C4_5Source(options.train)
        elif (options.data_format == 'Sparse'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Sparse Binary training data from', \
                      options.train
            source = SparseSource(options.train)
        else:
            raise Exception

        if options.algorithm == 'MaxEnt':
            print >>sys.stderr, 'Building MaxEnt model'
            cat = MaxEntInducer().train(source, \
                                        method='tao_lmvm', \
                                        quiet=options.quiet, \
                                        eps=options.smooth, \
                                        var=options.prior, \
                                        lasso=options.lasso, \
					max_it=options.max_iteration, \
                                        use_baseline=not options.no_baseline)
        elif options.algorithm == 'MaxEntRanker':
            print >>sys.stderr, 'Building MaxEnt model for ranking'
            cat = MaxEntRankerInducer().train(source, \
                                        method='tao_lmvm', \
                                        quiet=options.quiet, \
                                        eps=options.smooth, \
                                        var=options.prior, \
                                        lasso=options.lasso, \
					max_it=options.max_iteration, \
                                        use_baseline=not options.no_baseline)
        elif options.algorithm == 'Perceptron':
            print >>sys.stderr, 'Building Perceptron model'
            cat = MaxEntInducer().train(source, \
                                        method='perceptron', \
                                        quiet=options.quiet, \
                                        eps=options.smooth, \
                                        var=options.prior, \
                                        lasso=options.lasso, \
                                        use_baseline=not options.no_baseline)
        elif options.algorithm == 'NaiveBayes':
            print >>sys.stderr, 'Building naive Bayes model'
            cat =  NaiveBayesInducer().train(source, \
                                             quiet=options.quiet, \
                                             eps=options.smooth)

    # write model (if desired)

    if (options.train != '') and (options.model != ''):
        if (not options.quiet):
            print >> sys.stderr, 'Writing model to',options.model
        file = open(options.model,'w')
        cPickle.dump(cat,file,-1)
        file.close()

    # evaluate model

    if (options.test != ''):
        if (options.data_format == 'Simple'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Simple test data from', \
                      options.test
            source = SimpleSource(options.test)
        elif (options.data_format == 'Rainbow'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Rainbow test data from', \
                      options.test
            source = RainbowSource(options.test,binary=True)
        elif (options.data_format == 'C45'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading C4.5 test data from', \
                      options.test
            source = C4_5Source(options.test)
        elif (options.data_format == 'Sparse'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Sparse Binary test data from', \
                      options.test
            source = SparseSource(options.test)
        elif (options.data_format == 'SimpleRanker'):
            if (not options.quiet):
                print >> sys.stderr, 'Reading Simple Ranker test data from', \
                      options.test
            source = SimpleRankerSource(options.test)
        else:
            raise Exception

        acc = AccuracySink()
        cat.apply(source,acc)

        print 'Accuracy =', acc.score() * 100.0

        if options.confusion:
            acc.confusion()

        if options.display_predictions:
	    acc.predictions()

        if options.display_label_distributions:
	    acc.distributions()

    if (options.display_parameters):
        cat.dump()

    cat.cleanup()

if __name__ == '__main__':
    main()
     
