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

// perceptron estimation

// $Id: perceptron.cc,v 1.4 2006/04/20 14:48:10 jasonbaldridge Exp $

// Author: Jason Baldridge

#include "tadm.h"

#include <stdio.h>
#include <iostream>

using namespace std;

// perceptron
#define PERCEPTRON_DBL_MIN -1000000000000000000000.0

#undef __FUNCT__
#define __FUNCT__ "estimate_params_perceptron"
int 
estimate_params_perceptron(mle *e)
{
  cout << "Estimating parameters for perceptron." << endl;

  int ierr;

  // loop

  Vec accumulated_params;
  ierr = VecDuplicate(e->m->params, &accumulated_params); CHKERRQ(ierr);
  ierr = VecSet(accumulated_params, 0.0); CHKERRQ(ierr);
  ierr = VecAXPY(e->m->params, 1.0, accumulated_params); CHKERRQ(ierr);

  bool converged = false;
  int numTimesResultRepeated = 0;
  int previousNumCorrect = 0;

  int counter = 0;

  while (e->its++ < e->max_it && !converged) {

    cout << e->its << ": ";
    // update parameters
    double *weights;
    VecGetArray(e->m->params, &weights);

    double *counts;
    VecGetArray(e->d->p_ref, &counts);
    
    int numCorrect = 0, total = 0;
    for (int i=e->d->firstContext; i<=e->d->lastContext; i++) {

      const int start = e->d->context[i];
      const int end = e->d->context[i+1];

      int preferred = -1;
      int best = -1;
      double bestScore = PERCEPTRON_DBL_MIN;
      for (int j=start; j<end; j++) {

	if (counts[j] > 0)
	  preferred = j;

	int ncols;
	const int *columnIndices;
	const double *values;

	MatGetRow(e->d->data, j, &ncols, &columnIndices, &values);

	double score = 0;
	for (int k = 0; k < ncols; k++) {
	  score += weights[columnIndices[k]]*values[k];
	}

	MatRestoreRow(e->d->data, j, &ncols, &columnIndices, &values);

	if (score > bestScore) {
	  bestScore = score;
	  best = j;
	}

      }

      if (best > -1 && preferred > -1) {
	total++;
	if (best != preferred) {
	  int ncols;
	  const int *columnIndices;
	  const double *values;
	  MatGetRow(e->d->data, preferred, &ncols, &columnIndices, &values);

	  for (int k = 0; k < ncols; k++) {
	    weights[columnIndices[k]] += values[k];
	  }

	  MatRestoreRow(e->d->data, preferred, &ncols, &columnIndices, &values);
	  
	  MatGetRow(e->d->data, best, &ncols, &columnIndices, &values);

	  for (int k = 0; k < ncols; k++) {
	    weights[columnIndices[k]] -= values[k];
	  }

	  MatRestoreRow(e->d->data, best, &ncols, &columnIndices, &values);

	}

	else {
	  numCorrect++;
	}

      }
      else {
	cout << "No preferred item in range " 
	     << start << ".." << end << "! " 
	     << best << " :: " << preferred << endl;
      }
    }
    VecRestoreArray(e->d->p_ref, &counts);
    cout << numCorrect << "/" << total << endl;
    
    //double accuracy = static_cast<double>(numCorrect)/total;
    //cout << accuracy << " (" << numCorrect << "/" << total << ")" << endl;

    VecRestoreArray(e->m->params, &weights);
  
    VecAXPY(accumulated_params, 1.0, e->m->params);

    //VecView(e->m->params, PETSC_VIEWER_STDOUT_SELF);

    if (total == numCorrect) {
      converged = true;
    }

    if (numCorrect == previousNumCorrect) {
      numTimesResultRepeated++;
    } else {
      previousNumCorrect = numCorrect;
      numTimesResultRepeated = 0;
    }

    if (numTimesResultRepeated > 2) {
      converged = true;
    }

    counter++;

  }

  VecScale(accumulated_params, 1.0/counter);
  
  VecCopy(accumulated_params, e->m->params);

  //VecView(e->m->params, PETSC_VIEWER_STDOUT_SELF);
  //VecView(accumulated_params, PETSC_VIEWER_STDOUT_SELF);

  // Finish up
  return 0;
}
