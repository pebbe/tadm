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

// fileio.cc
//
// $Id: fileio.cc,v 1.3 2007/09/03 19:16:48 jasonbaldridge Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"
#include "fileio.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

// refill input buffer

#undef __FUNCT__
#define __FUNCT__ "Datafile::fillBuffer"
int
Datafile::fillBuffer()
{
  int len = gzread(file,buffer,BUFF_LEN);

  buffend = buffer + len - 1;
  buffptr = buffer;

  return len;
}

// get next line of input

#undef __FUNCT__
#define __FUNCT__ "Datafile::getLine"
char *
Datafile::getLine(char *str, int size)
{
  char *ptr = str, c;
  int len = 1;

  if ((c=getChar()) == (char)EOF)
    return NULL;

  while ((c != '\n') && (c != (char)EOF)) {
    *ptr++ = c;
    if (++len >= size) {
      cerr << "Line overflow -- data lost" << endl;
      break;
    }
    c = getChar();
  }

  *ptr++ = '\0';

  return str;
}

// get next character of input

#undef __FUNCT__
#define __FUNCT__ "getChar"
char 
Datafile::getChar()
{
  // refill buffer, if necessary
  if (buffptr > buffend) 
    if (fillBuffer() <= 0)
      return EOF;
  
  return *buffptr++;
}

// unget next character of input

#undef __FUNCT__
#define __FUNCT__ "Dataset::getChar"
char 
Datafile::unGetChar()
{
  if (buffptr == buffer) {
    cerr << "Buffer underflow -- data lost" << endl;
  } else {
    buffptr--;
  }

  return 1;
}


// open a datafile for reading

#undef __FUNCT__
#define __FUNCT__ "Datafile::Datafile"
Datafile::Datafile(const char *filename)
{
  file = gzopen(filename,"rb");

  if (!file) 
    cerr << "Can't open event input file" << endl;
}


// close a datafile 

#undef __FUNCT__
#define __FUNCT__ "Datafile::~Datafile"
Datafile::~Datafile()
{
  gzclose(file);
}

// find first context in datafile

#undef __FUNCT__
#define __FUNCT__ "Datafile::firstContext"
void 
Datafile::firstContext()
{
  // rewind
  
  gzseek(file, 0L, SEEK_SET);
  fillBuffer();

  // Skip header

  if (getChar() == '&') {
    while (getChar() != '/') ;
    while (getChar() != '\n') ;
  } else {
    unGetChar();
  }
}

// skip to next line in datafile

#undef __FUNCT__
#define __FUNCT__ "Datafile::skipLine"
void 
Datafile::skipLine()
{
  // Skip to next line
  while (getChar() != '\n') ;
}

// get event count for next context

#undef __FUNCT__
#define __FUNCT__ "Datafile::getCount"
int 
Datafile::getCount(int *count)
{
  if ((lineptr = getLine(line, LINE_LEN))) {
    *count = strtol(lineptr, NULL, 10);
    return 0;
  } 
  
  return EOF;
  
}

// get frequency and feature count for next event

#undef __FUNCT__
#define __FUNCT__ "Datafile::getFreq"
int 
Datafile::getFreq(double *freq, int *count)
{
  if ((lineptr = getLine(line,LINE_LEN))) {
    *freq = strtod(lineptr,&lineptr);
    *count = strtol(lineptr,&lineptr,10);
    return 0;
  } 
  
  return EOF;
}

// get frequency, prior, and feature count for next event

#undef __FUNCT__
#define __FUNCT__ "Datafile::getFreq"
int 
Datafile::getFreq(double *freq, double *prior, int *count)
{
  if ((lineptr = getLine(line,LINE_LEN))) {
    *freq = strtod(lineptr,&lineptr);
    *prior = strtod(lineptr,&lineptr);
    *count = strtol(lineptr,&lineptr,10);
    return 0;
  } 
  
  return EOF;
}

// get next freature/value pair

#undef __FUNCT__
#define __FUNCT__ "Datafile::getPair"
int 
Datafile::getPair(int *feat, double *val)
{
  if (lineptr) {
    *feat = strtol(lineptr, &lineptr, 10);
    *val = strtod(lineptr, &lineptr);
    return 0;
  } 
  
  return EOF;
}
