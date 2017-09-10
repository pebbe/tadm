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

"""Categorizer objects that assign scores to categories using previously trained parameters."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "12 February 2004"
__version__ = "$Revision: 1.4 $"

import math
import os
import tempfile

class Model:
    pass


class MaxentModel(Model):

    def __init__(self, eventfile, paramsfile):
        self.eventfile = eventfile
        self.paramsfile = paramsfile

    def cleanup(self):
        os.unlink(self.eventfile)
        os.unlink(self.paramsfile)
        return


class Ranker(Model):

    def __init__(self):
        self.feature_map = {}
        
    def rank(self, candidates):
        best_score = -1000000.0
        scores = ()
        best_index = -1
        index = 0

	scores = []
	total = 0

        for label,features_and_counts in candidates:
            score = 0.0
            for feat,count in features_and_counts:
                score += self.feature_map.get(feat,0.0)*count

            if (score > best_score):
                best_score = score
                best_index = index

            index += 1

	    score = math.exp(score)
	    scores.append(score)
	    total += score

	distribution = [x/float(total) for x in scores]

        return best_index, distribution

    def apply(self,source,sink):

        for gold_index, num_candidates, candidates, line in source:
            if num_candidates > 1:
		best, distribution = self.rank(candidates)
                sink.update(line, gold_index, str(best), distribution)

    def dump(self):

        for feat in self.feature_map:
            print feat,self.feature_map[feat]
             
class MaxEntRanker(Ranker, MaxentModel):
    
    def __init__(self, eventfile=tempfile.mktemp(), paramsfile=tempfile.mktemp()):
        Ranker.__init__(self)
        MaxentModel.__init__(self, eventfile, paramsfile)


class Categorizer(Model):

    def __init__(self):
        self.categories = {}
        self.features = {}
	self.smooth_m = 0
    
    def categorize(self,event):
        best_score = -1000000.0
        cat = ''
	total = 0
	scores = []
	labels = self.categories.keys()
        for cand in labels:
            score = self.features.get((cand),0.0)
            for f,v in event:
                score += self.features.get((f,v,cand),0.0)
            if (score > best_score):
                best_score = score
                cat = cand
	    score = math.exp(score)
	    scores.append(score)
	    total += score

	scores = [x/float(total) for x in scores]
	distribution = zip(labels,scores)

        return cat, distribution

    def apply(self,source,sink):
        for feats,cl,line in source:
            if (cl == None):
                sink.update(line,cl,None)
            else:
		cat, distribution = self.categorize(feats)
                sink.update(line,cl, cat, distribution)

    def dump(self):

        for feat in self.features:
            print feat,self.features[feat]

                         
class MaxEntCategorizer(Categorizer,MaxentModel):

    def __init__(self, eventfile=tempfile.mktemp(), paramsfile=tempfile.mktemp()):
        Categorizer.__init__(self)
        MaxentModel.__init__(self, eventfile, paramsfile)

class NaiveBayesCategorizer(Categorizer):
    pass

