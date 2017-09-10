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

"""Objects that can be used to call TADM in order to train the parameters."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "12 February 2004"
__version__ = "$Revision: 1.12 $"

from __future__ import division
from math import log
import os
import string
import sys
import tempfile
from tadm.model import *

class Inducer:
    pass

class MaxEntInducer(Inducer):

    def train(self,source,method='tao_lmvm',eps=0,frtol=1e-8,var=0,lasso=0,\
              quiet=False, max_it=0,use_baseline=False,cutoff=0):
        """Construct a MaxEnt categorizer from a data source"""

        model = MaxEntCategorizer()

        # Collect features

        counts = {}
        baseline = None
                
        for feats,cl,line in source:
            if cl == None:
                continue
            model.categories[cl] = model.categories.get(cl,0) + 1
            if use_baseline and baseline == None:
                baseline = cl
            for f,v in feats:
                counts[(f,v)] = counts.get((f,v),0) + 1
                    
        # collect category statistics
        
        n = sum(model.categories.values())
        for cl in model.categories:
            model.categories[cl] /= n
            
        # number features
        
        feats = 0

        for cl in model.categories:
            if (baseline != cl):
                model.features[(cl)] = feats
                feats += 1
	    for (f,v) in counts:
		if (counts[(f,v)] > cutoff):
		    model.features[(f,v,cl)] = feats
		    feats += 1

        # generate events file

        self.make_events(source,model,eps)
        
        # run tadm

        varopt = ''
        if (var != 0):
            varopt = '-l2 %d ' % (var)
            # varfile=tempfile.mktemp()
            # varopt = '-variances %s' % (varfile)
            # out = open(varfile,'w')
            # for i in xrange(0,feats):
            #     print 1.0/var
            # out.close()
        elif (lasso != 0):
            varopt = '-l1 %d ' % (lasso)
            
        maxit_opt = ''
	if (max_it != 0):
	    maxit_opt = '-max_it %d ' % (max_it)
                            
        command = 'tadm -monitor -method %s -events_in %s -params_out %s %s %s' % (method,model.eventfile,model.paramsfile,varopt,maxit_opt)

        if (quiet):
            os.system(command+' >/dev/null')
        else:
            print >> sys.stderr,'======='
            print >> sys.stderr,command
            os.system(command)
            print >> sys.stderr,'======='
        
        # load parameter values
                
        weights = [ float(w.rstrip())
                    for w in open(model.paramsfile).xreadlines()]

        model.count = 0
        for i in xrange(0,len(weights)):
            if (weights[i] != 0):
                model.count = model.count + 1

        for f,v in model.features.items():
            if (v < len(weights)):
                model.features[f] = weights[v]
            else:
                model.features[f] = 0.0
                
        # clean up temporary files

        #os.unlink(self.eventfile)
        #os.unlink(self.paramsfile)

        return model


    def make_events(self,source,model,eps=0):

        events = {}

        # read events

        for feats,cl,line in source:
            if cl == None:
                continue
            feats = tuple(feats)
            try:
                entry = events[feats]
                entry[cl] = entry.get(cl,0) + 1
            except KeyError:
                events[feats] = { cl : 1 }
  
        # output events
  
        outfile = open(model.eventfile,'w')
        cl_list  = model.categories.keys()

        for feats, entry in events.items():

            print >> outfile, len(cl_list)
            for c in cl_list:
                codes = []
                if (model.features.has_key((c))):
                    codes.append(model.features[(c)])
                for f,v in feats:
                    if (model.features.has_key((f,v,c))):
                        codes.append(model.features[(f,v,c)])
		    else:
			print f,v,c

                print >> outfile, entry.get(c,0)+eps, len(codes),
                for f in codes:
                    print >> outfile, f, '1',
                print >> outfile

        outfile.close()
        
        return
 

class MaxEntRankerInducer(Inducer):

    def train(self,source,method='tao_lmvm',eps=0,frtol=1e-8,var=0,lasso=0,\
              quiet=False, max_it=0, use_baseline=False,cutoff=0):
        """Construct a MaxEnt ranker from a data source"""
        
        model = MaxEntRanker()

        # Collect and index the features
        feature_counts = {}
        feature_index = 0
        for gold_index, num_candidates, candidates, line in source:
            for label,features_and_counts in candidates:
                if label == None:
                    print "Null label??", features
                    continue
                for f,c in features_and_counts:
                    index = None
                    if model.feature_map.has_key(f):
                        index = model.feature_map[f]
                    else:
                        index = feature_index
                        feature_index += 1
                        model.feature_map[f] = feature_index

                    feature_counts[index] = feature_counts.get(index,0) + c
                
        # generate events file

        self.make_events_rank(source,model,eps)
        
        # run tadm

        varopt = ''
        if (var != 0):
            varopt = '-l2 %d ' % (var)
            # varfile=tempfile.mktemp()
            # varopt = '-variances %s' % (varfile)
            # out = open(varfile,'w')
            # for i in xrange(0,feats):
            #     print 1.0/var
            # out.close()
        elif (lasso != 0):
            varopt = '-l1 %d ' % (lasso)

        maxit_opt = ''
	if (max_it != 0):
	    maxit_opt = '-max_it %d ' % (max_it)
                            
        command = 'tadm -monitor -method %s -events_in %s -params_out %s %s %s' % (method,model.eventfile,model.paramsfile,varopt,maxit_opt)

        if (quiet):
            os.system(command+' >/dev/null')
        else:
            print >> sys.stderr,'======='
            print >> sys.stderr,command
            os.system(command)
            print >> sys.stderr,'======='
        
        # load parameter values
                
        weights = [ float(w.rstrip())
                    for w in open(model.paramsfile).xreadlines()]

        model.count = 0
        for i in xrange(0,len(weights)):
            if (weights[i] != 0):
                model.count = model.count + 1

        for f,v in model.feature_map.items():
            if (v < len(weights)):
                model.feature_map[f] = weights[v]
            else:
                model.feature_map[f] = 0.0
                
        return model

    def make_events_rank(self,source,model,eps=0):

        # output events
  
        outfile = open(model.eventfile,'w')

        for gold_index, num_candidates, candidates, line in source:
            print >> outfile, num_candidates
            for label,features_and_counts in candidates:
                print >> outfile, label, len(features_and_counts),
                for f,c in features_and_counts:
                    if model.feature_map.has_key(f):
                        print >> outfile, model.feature_map[f], c,
                print >> outfile

        outfile.close()
        
        return


class NaiveBayesInducer(Inducer):

    def train(self,source,eps=0,quiet=False):
        """Construct a Naive Bayes categorizer from a data source"""

        model = NaiveBayesCategorizer()

	use_smoothing = 0
	if eps > 0:
	    use_smoothing = 1

	model.smooth_m = eps
	    
        # count features

        freq = {}
        features = {}
                
        for feats,cl,line in source:
            if cl == None:
                continue
            model.categories[cl] = model.categories.get(cl,0) + 1
            for f,v in feats:
                features[(f,v)] = 1
                freq[(f,v,cl)] = freq.get((f,v,cl),0) + 1
            
        # compute parameter values


        overflow = False
        for (f,v) in features:
            for cl in model.categories:
                try:
                    model.features[(f,v,cl)] = \
			log(1*use_smoothing+freq.get((f,v,cl),0)) - \
			log(eps*use_smoothing+model.categories[cl])
                except OverflowError:
                    overflow = True

        if overflow:
            print >>sys.stderr, 'Zero probabilities found -- try smoothing counts (-s)!'                    
            
        # collect category priors
        
        n = sum(model.categories.values())
        for cl in model.categories:
            model.features[(cl)] = log(model.categories[cl]) - log(n)

        return model
