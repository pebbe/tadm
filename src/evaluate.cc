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

// ***
// *** evaluate
// ***

// $Id: evaluate.cc,v 1.2 2007/09/11 21:06:23 jasonbaldridge Exp $

//  Copyright (c) 2002 Robert Malouf

#include <iostream>
#include <fstream>
#ifdef __GNUC__
#include <ext/slist>
#else
#include <slist>
#endif

#include <cmath>
#include <unistd.h>
#include "fileio.h"

#define DBL_MIN -1000000000000000000000.0

using namespace std;

#ifdef __GNUC__
using namespace __gnu_cxx;
#endif

int main(int argc, char **argv)
{

  // check command line arguments

  ofstream scoresStream;

  char c;
  bool error = false, divergence = false, accuracy = false;
  double corr = 0.0;
  double thresh = 0.0;

  while ((c = getopt(argc, argv, "ac:dt:s:")) != EOF)
  {
    switch (c)
    {
    case 't':
    {
      thresh = atof(optarg);
      break;
    }
    case 'c':
    {
      ifstream tmp(optarg);
      tmp >> corr;
      cerr << "Correction factor = " << corr << endl;
      tmp.close();
      break;
    }
    case 'd':
      divergence = true;
      break;
    case 'a':
      accuracy = true;
      break;
    case 's':
    {
      scoresStream.open(optarg, ios_base::out);
      break;
    }
    default:
      error = true;
    }
  }

  if (error || (argc - optind != 2))
  {
    cerr << "Usage: " << argv[0] << " [-t thresh] [-s scores ] [-c n] [-d] [-a] <pfile> <events>" << endl;
    return 1;
  }

  // count parameters

  ifstream pfile(argv[optind]);
  if (!pfile)
  {
    cerr << "Can't open parameter file!" << endl;
    return -1;
  }

  int feat = 0;
  double t;

  while (true)
  {
    pfile >> t;
    if (pfile.eof())
      break;
    feat++;
  }

  // read parameters

  double params[feat];

  pfile.clear();
  pfile.seekg(0, ios::beg);

  int f = 0;

  for (int i = 0; i < feat; ++i)
  {
    pfile >> params[i];
    double p = fabs(params[i]);
    if (p > thresh)
      f++;
    else
      params[i] = 0.0;
  }

  pfile.close();

  // read and score event file

  double score = 0.0;
  int contexts = 0, events;
  slist<pair<double, double>> scores;

  Datafile in(argv[optind + 1]);
  in.firstContext();

  while (in.getCount(&events) != EOF)
  {

    // read and score events

    if (events > 0)
    {

      double maxprob = DBL_MIN, maxfreq = DBL_MIN, freq;
      double totalprob = 0.0, totalfreq = 0.0;
      int count;

      // std::cout << std::endl;

      contexts++;

      // read events for context

      for (int i = 0; i < events; i++)
      {

        double vv = 0.0, prob = 0.0;

        // read event
        if (in.getFreq(&freq, &count) == EOF)
          cerr << "Error reading data file" << endl;
        for (int j = 0; j < count; j++)
        {
          int f;
          double v;
          if (in.getPair(&f, &v) == EOF)
            cerr << "Error reading data file" << endl;
          if (f >= feat)
            continue;
          vv += v;
          prob += v * params[f];
        }

        // add correction feature
        if (corr != 0.0)
          prob += (corr - vv) * params[feat - 1];

        //
        // when per-event scoring is active, record the (unnormalized) current
        // event score.
        //
        if (scoresStream.is_open())
        {
          if (i > 0)
            scoresStream << ' ';
          scoresStream << prob;
        } // if

        // remember event
        scores.push_front(make_pair(freq, prob));
        totalprob += exp(prob);
        totalfreq += freq;
        if (freq > maxfreq)
          maxfreq = freq;
        if (prob > maxprob)
          maxprob = prob;
      }

      //
      // to preserve file format compatibility with `score' outputs, put in a
      // newline after each context.
      //
      if (scoresStream.is_open())
        scoresStream << endl;

      if (divergence)
      {

        // get divergence for context

        while (!scores.empty())
        {

          const pair<double, double> &it = scores.front();

          const double p = it.first / totalfreq;
          const double q = exp(it.second) / totalprob;

          if (totalfreq > 0)
            if (p != 0.0 && q != 0.0)
              score += p * (log(p) - log(q));

          scores.pop_front();
        }
      }
      else
      {

        // get score for context

        double best = 0.0, total = 0.0;

        while (!scores.empty())
        {

          const pair<double, double> &it = scores.front();
          if (it.second == maxprob)
          {
            total++;
            if (accuracy)
              best += it.first;
            else if (it.first == maxfreq)
              best++;
          }

          scores.pop_front();
        }

        score += best / total;
        // std::cout << best << " " << total << " " << score << std::endl;
      }
    }
  }

  if (divergence)
  {
    cout << "DIV: " << argv[optind] << " " << argv[optind + 1] << " " << score << " " << f << endl;
  }
  else
  {
    if (accuracy)
      cout << "ACC: ";
    else
      cout << "SCORE: ";
    cout << argv[optind] << " " << argv[optind + 1] << " " << score / (double)contexts * 100.0 << " " << f << endl;
  }
}
