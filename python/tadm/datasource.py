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

"""Classes for reading training and testing material from different types of data sources."""

__author__ = "Rob Malouf <rmalouf@mail.sdsu.edu>"
__date__ = "12 February 2004"
__version__ = "$Revision: 1.4 $"

class Source:
    pass

class C4_5Source(Source):
    """Data source class for C4.5 files"""

    def __init__(self,infile):
        """Create a data source from a C4.5 file"""
        self.stream = open(infile)

        return

    def __iter__(self):
        return self 

    def next(self):

        # If we're at the end, rewind stream and stop

        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration

        # get features

        if (line[-1] == '.'):
            line = line[:-1]
        items = line.split(',')

        # get category
        
        cl = intern(items.pop())

        # collect feature/value pairs

        j = 0        
        for i in xrange(0,len(items)):
            #items[i] = (j,intern(items[i]))
            items[i] = (0,intern(items[i]))
            #print items[i]
            j += 1
        return (items,cl,line)


class SimpleSource(Source):
    """Data source class for simple format files"""

    def __init__(self,infile):
        """Create a data source from a simple sourc file"""
        self.stream = open(infile)
        return

    def __iter__(self):
        return self 

    def next(self):

        # If we're at the end, rewind stream and stop

        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration

        # get features

        items = line.split()

        # get category

        cl = intern(items[0])
        items = items[1:]

        # collect feature/value pairs

        for i in xrange(0,len(items)):
            items[i] = (0,intern(items[i]))

        return (items,cl,line)


class SimpleRankerSource(Source):
    """Data source class for simple format files used for ranking"""

    def __init__(self,infile):
        """Create a data source from a simple source file"""
        self.stream = open(infile)
        return

    def __iter__(self):
        return self 

    def next(self):

        # If we're at the end, rewind stream and stop

        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration

        # get number of candidates
        num_candidates = int(line.strip())


        # pull in each candidate and index, etc

        candidates = []
        gold_index = -1
        for i in range(0,num_candidates):

            # read a line
            items = self.stream.readline().rstrip().split()

            # get label
            label = intern(items[0])
            if label == "1":
                gold_index = i

            # collect feature/count pairs 
            items = items[1:]
            features_and_counts = []
            for i in xrange(0,len(items)):
                fc_pair = (intern(items[i]), 1)
                features_and_counts.append(fc_pair)

            candidates.append((label,features_and_counts))

        return (str(gold_index), num_candidates, candidates, line)


class RainbowSource(Source):
    """Data source class for files generated using rainbow -B sin"""

    binary = False
    def __init__(self,infile,binary=False):
        """Create a data source from a rainbow file"""
        self.stream = open(infile)
        self.binary = binary
        return

    def __iter__(self):
        return self 

    def next(self):

        line = self.stream.readline().rstrip()
        items = re.split('\s+',line)
        
        # if we're at the end, rewind stream and stop

        if (items == ['']):
            self.stream.seek(0)
            raise StopIteration

        # get category 

        cl = intern(items[1])

        # collect feature/value pairs

        pairs = []
        for i in xrange(2,len(items),2):
            if (self.binary):
                pairs.append((items[i],1))
            else:
                pairs.append((items[i],items[i+1]))

        return (pairs,cl,line)

class SparseSource(Source):
    """Data source class for Sparse Binary files"""

    def __init__(self,infile):
        """Create a data source from a Sparse Binary file"""
        self.stream = open(infile)
        return

    def __iter__(self):
        return self 

    def next(self):

        # if we're at the end, rewind stream and stop

        line = self.stream.readline().rstrip()
        if (line == ''):
            self.stream.seek(0)
            raise StopIteration

        # get features

        if (line[-1] == '.'):
            line = line[:-1]
        items = line.split(',')

        # get category (minus trailing period)

        cl = intern(items.pop())
        if (cl[-1] == '.'):
            cl = intern(cl[:-1])

        # collect feature/value pairs

        for i in xrange(0,len(items)):
            items[i] = (items[i],1)
        return (items,cl,line)
