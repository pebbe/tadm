/////////////////////////////////////////////////////////////////////////////////
// The Toolkit for Advanced Discriminative Modeling
// Copyright (C) 2001-2005 Robert Malouf
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
/////////////////////////////////////////////////////////////////////////////////

// fileio.h
//
// $Id: fileio.h,v 1.1.1.1 2005/08/10 16:10:39 jasonbaldridge Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#ifndef _FILEIO_H_
#define _FILEIO_H_

#include "zlib.h"

const int LINE_LEN = 100000*4;   // Maximum line length in data file
const int BUFF_LEN=100000*16;

class Datafile {

  gzFile file;                      // file pointer
  char line[LINE_LEN], *lineptr;
  char buffer[BUFF_LEN], *buffptr, *buffend;
  
  int fillBuffer();                 // refill input buffer

  char *getLine(char *str, int size); 
  char getChar();
  char unGetChar();

 public:

  Datafile(const char *filename);   // constructor (open dataset)
  ~Datafile();                      // destructor (close dataset)

  void firstContext();              // find first context in datafile
  void skipLine();                  // skip to next line in datafile

  int getCount(int *count);
  int getFreq(double *freq, int *count); 
  int getFreq(double *freq, double *prior, int *count);
  int getPair(int *feat, double *val);
};

#endif /* _FILEIO_H_ */    
