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

// maximum likelihood estimation

// $Id: iis.cc,v 1.1.1.1 2005/08/10 16:10:39 jasonbaldridge Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"

const int NEWTON_MAX = 10;

// compute parameter updates for IIS

#undef __FUNCT__
#define __FUNCT__ "update"
int
update (mle *e, Vec delta, Vec fsharp, int maxm, Mat datat, const PetscInt *ic, 
	const double *vv, double *a)
{
  PetscInt ncols, flops;
  PetscScalar *xe, *xq, *xf;

  // set up

  PetscLogEventBegin(UPDATE_EVENT, 0, 0, 0, 0);

  VecGetArray(e->d->e_ref, &xe);
  VecGetArray(e->m->q, &xq);
  VecGetArray(fsharp, &xf);

  flops = 0;

  // compute updates

  for (int i = 0; i < e->d->nFeats; i++) {
     
    double oldb, b;
    int ii;

    // set up coefficients 
    
    for (int j = 0; j < maxm; j++)
      a[j] = 0.0;

    MatGetRow(datat, i, &ncols, &ic, &vv);

    for (int j = 0; j < ncols; j++) 
      a[maxm - (int)xf[ic[j]]] += xq[ic[j]] * vv[j];

    MatRestoreRow(datat, i, &ncols, &ic, &vv);
     
    flops += 2*ncols;
     
    // set up for iteration
     
    oldb = 1e8;
    b = 1.0;
    ii = 1;
     
    // Newton-Raphson estimation
     
    while ((fabs(b-oldb) >= 0.0001) && (ii <= NEWTON_MAX)) {
        
      double f, df;

      oldb = b;
        
      f = a[0];
      df = 0.0;
        
      for (int j = 1; j < maxm; j++) {
	df = df * b + f;
	f = f * b + a[j];
      }
        
      df = df * b + f;
      f = f * b + xe[i];

      // penalty goes here

      flops += maxm*4;
        
      if (df != 0) {
	b = b - f / df;
	flops += 2;
      }
        
      ii++;

    }        
     
    // set update
     
    b = log(b);
    flops++;
    VecSetValue(delta, i, b, INSERT_VALUES);
  }
     
  // wrap up

  VecAssemblyBegin(delta);
  VecAssemblyEnd(delta);

  VecRestoreArray(e->m->q, &xq);
  VecRestoreArray(e->d->e_ref, &xe);
  VecRestoreArray (fsharp, &xf);
  PetscLogFlops(flops);
  PetscLogEventEnd(UPDATE_EVENT, 0, 0, 0, 0);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "estimate_params_iis"
int 
estimate_params_iis(mle *e)
{
  int maxm, ierr, i, *ic;
  double *a, *vv, f;
  const double one = 1.0;

  Vec tmp, fsharp, delta;

  if (e->d->penalty)
    SETERRQ(PETSC_ERR_SUP,"IIS for penalized likelihood not implemented");

  PetscLogEventBegin(IIS_EVENT, 0, 0, 0, 0);

  // set up fsharp

  ierr = VecDuplicate(e->m->params, &tmp);CHKERRQ(ierr);
  ierr = VecSet(tmp,1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(e->m->q, &fsharp);CHKERRQ(ierr);

  ierr = MatMultTranspose(e->d->data, tmp, fsharp);CHKERRQ(ierr);

  ierr = VecMax(fsharp, &i, &f);CHKERRQ(ierr);
  maxm = (int)f;

  ierr = VecDestroy(tmp);CHKERRQ(ierr);

  // allocate temporary space

  ierr = PetscMalloc(maxm*sizeof(double), &a);CHKERRQ(ierr);
  ierr = PetscMalloc(e->d->nClasses*sizeof(double), &vv);CHKERRQ(ierr);
  ierr = PetscMalloc(e->d->nClasses*sizeof(int), &ic);CHKERRQ(ierr);
  ierr = VecDuplicate(e->m->params, &delta);CHKERRQ(ierr);

  // initialize q

  MatMultTranspose(e->d->data, e->m->params, e->m->q); 
  VecExp(e->m->q);
  normalize(e->m->q, e->d);
  log_likelihood(e->d->p_ref, e->m->q, &f, &(e->m->h));
  e->m->kl = f - e->d->h0;
  e->its = 0;
  e->fg++;

  // start logging

  if (e->monitor) {
    PetscPrintf(PETSC_COMM_WORLD,
		"\n It       D(p_ref||p)              H(p)\n\n");
    PetscPrintf(PETSC_COMM_WORLD,"%4d%18.8e%18.8e\n", 0, e->m->kl, e->m->h);
  }

  // loop

  while (!maxent_conv(e->its, f, e)) {
    
    // update parameters

    update(e, delta, fsharp, maxm, e->d->data, ic, vv, a);
    VecAXPY(e->m->params,1.0,delta);

    // recalculate q

    MatMultTranspose(e->d->data, e->m->params, e->m->q); 
    VecExp(e->m->q);
    normalize(e->m->q, e->d);
    log_likelihood(e->d->p_ref, e->m->q, &f, &(e->m->h));
    e->m->kl = f - e->d->h0;
    e->its++;
    e->fg++;

    if (e->monitor) {
      PetscPrintf(PETSC_COMM_WORLD,"%4d%18.8e%18.8e\n", e->its, e->m->kl, 
		  e->m->h);
    }
  }

  // Finish up

  ierr = VecDestroy(delta);CHKERRQ(ierr);
  ierr = VecDestroy(fsharp);CHKERRQ(ierr);

  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(vv);CHKERRQ(ierr);
  ierr = PetscFree(ic);CHKERRQ(ierr);

  PetscLogEventEnd(IIS_EVENT, 0, 0, 0, 0);

  return 0;
}

// generalized iterative scaling

#undef __FUNCT__
#define __FUNCT__ "estimate_params_gis"
int 
estimate_params_gis(mle *e)
{
  int ierr;
  double *xe, *xu, *xq, f;
  const double zero = 0.0, one = 1.0, scale = 1.0/(e->d->c);

  Vec delta,eq;

  if (e->d->penalty)
    SETERRQ(PETSC_ERR_SUP,"GIS for penalized likelihood not implemented");

  PetscLogEventBegin(IIS_EVENT, 0, 0, 0, 0);

  // allocate temporary space

  ierr = VecDuplicate(e->m->params, &delta);CHKERRQ(ierr);
  ierr = VecDuplicate(e->m->params, &eq);CHKERRQ(ierr);

  // set up

  ierr = VecSet(e->m->params,0.0);CHKERRQ(ierr);  

  // initialize q

  MatMult(e->d->data, e->m->params, e->m->q); 
  VecExp(e->m->q);
  normalize(e->m->q, e->d);
  MatMultTranspose(e->d->data,e->m->q,eq);
  log_likelihood(e->d->p_ref, e->m->q, &f, &(e->m->h));
  e->m->kl = f - e->d->h0;
  e->its = 0;
  e->fg++;

  // start logging

  if (e->monitor) {
    PetscPrintf(PETSC_COMM_WORLD,
		"\n It       D(p_ref||p)              H(p)\n\n");
    PetscPrintf(PETSC_COMM_WORLD,"%4d%18.8e%18.8e\n", 0, e->m->kl, e->m->h);
  }

  // loop

  VecGetArray(e->d->e_ref,&xe);

  while (!maxent_conv(e->its, f, e)) {

    // update parameters
 
    VecGetArray(delta,&xu);
    VecGetArray(eq,&xq);
 
    for (int i=0; i<e->d->nFeats; i++) {
      if (xe[i] != 0.0) {
        xu[i] = scale*(log(-xe[i])-log(xq[i]));
      } else {
        xu[i] = 0.0;
      }
    }
 
    VecRestoreArray(delta,&xu);
    VecRestoreArray(eq,&xq);

    VecAXPY(e->m->params,1.0,delta);

    // recalculate q

    MatMult(e->d->data, e->m->params, e->m->q); 
    VecExp(e->m->q);
    normalize(e->m->q, e->d);
    MatMultTranspose(e->d->data,e->m->q,eq);
    log_likelihood(e->d->p_ref, e->m->q, &f, &(e->m->h));
    e->m->kl = f - e->d->h0;
    e->its++;
    e->fg++;


    if (e->monitor) {
      PetscPrintf(PETSC_COMM_WORLD,"%4d%18.8e%18.8e\n", e->its, e->m->kl, 
		  e->m->h);
    }
  }

  // Finish up

  VecRestoreArray(e->d->e_ref,&xe);
  ierr = VecDestroy(delta);CHKERRQ(ierr);

  PetscLogEventEnd(IIS_EVENT, 0, 0, 0, 0);

  return 0;
}
