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

"""Result sink that collects statistics about results of using classifier."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "12 February 2004"
__version__ = "$Revision: 1.6 $"

from __future__ import division

class ResultSink:
    pass

class AccuracySink(ResultSink):
    """Result sink that collects accuracy statistics"""

    def __init__(self):
        self.total = 0
        self.right = 0
        self.matrix = {}
        self.tags = {}
	self.model_predictions = []
	self.label_distributions = []

    def update(self,feats,cl,best,label_distribution):
        if (best != None):
            self.total += 1
            if (cl == best):
                self.right += 1
            self.tags[cl] = 1
            self.tags[best] = 1
            self.matrix[(cl,best)] = self.matrix.get((cl,best),0) + 1
	self.model_predictions.append((cl,best))
	self.label_distributions.append(label_distribution)

    def score(self):
        """Returns accuracy so far"""
        return self.right / self.total

    def confusion(self):
        """Print confusion matrix"""
        print 'Confusion matrix'
        print 
        print '\tas'
        print 'is\t',"\t".join(self.tags.keys())
        for tag1 in self.tags.keys():
            print tag1,'\t',
            for tag2 in self.tags.keys():
                print self.matrix.get((tag1,tag2),0),'\t',
            print            

    def predictions(self):
        truth_and_predicted = [str(x)+"\t"+y for (x,y) in self.model_predictions]
	print "====== Model predictions on test data ======"
        print "True\tPredicted"
	#print "\n".join(self.model_predictions)
	print "\n".join(truth_and_predicted)
	print "============================================"

    def distributions(self):
	print "====== Model distributions on test data ======"
	print "\n".join([repr(item) for item in self.label_distributions])
	print "============================================"
