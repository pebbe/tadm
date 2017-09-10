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

// $Id: probs.cc,v 1.4 2007/09/03 19:16:48 jasonbaldridge Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"
#include "version.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>

// log-likelihood and entropy
//   LL(p,q) = - sum_i p(x_i) log q(x_i)
//   H(q) = - sum_i q(x_i) log q(x_i)

#undef __FUNCT__
#define __FUNCT__ "log_likelihood"
int log_likelihood(Vec p, Vec q, double *f, double *h)
{
  PetscScalar ff = 0.0, hh = 0.0, *pp, *qq;
  int n, ierr;

  ierr = PetscLogEventBegin(ENTROPY_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  ierr = VecGetLocalSize(p, &n);
  CHKERRQ(ierr);

  ierr = VecGetArray(p, &pp);
  CHKERRQ(ierr);
  ierr = VecGetArray(q, &qq);
  CHKERRQ(ierr);

  for (int i = 0; i < n; i++)
  {
    if (qq[i] != 0.0)
    {
      double t = log(qq[i]);
      ff -= pp[i] * t;
      hh -= qq[i] * t;
    }
  }

  ierr = VecRestoreArray(p, &pp);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(q, &qq);
  CHKERRQ(ierr);

  ierr = PetscLogFlops(n * 7.0);
  CHKERRQ(ierr);

  MPI_Allreduce(&ff, f, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Allreduce(&hh, h, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  ierr = PetscLogEventEnd(ENTROPY_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return 0;
}

// calculate the entropy of a probability distribution:
//   H(p)= -sum_i p(x_i) log p(x_i)

#undef __FUNCT__
#define __FUNCT__ "entropy"
double
entropy(Vec x)
{
  PetscScalar h, hh = 0.0, *xx;
  int n, ierr;

  ierr = PetscLogEventBegin(ENTROPY_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  ierr = VecGetLocalSize(x, &n);
  CHKERRQ(ierr);

  ierr = VecGetArray(x, &xx);
  CHKERRQ(ierr);

  for (int i = 0; i < n; i++)
    if (xx[i] != 0.0)
      hh -= xx[i] * log(xx[i]);

  ierr = VecRestoreArray(x, &xx);
  CHKERRQ(ierr);

  ierr = PetscLogFlops(n * 3.0);
  CHKERRQ(ierr);

  MPI_Allreduce(&hh, &h, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  ierr = PetscLogEventEnd(ENTROPY_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return h;
}

// sum unnormalized probabilities for each context

#undef __FUNCT__
#define __FUNCT__ "sum_z"
int sum_z(Vec p, Dataset *d)
{
  int ierr, high, low, nprocs;
  PetscScalar *xp, *zz;

  ierr = PetscLogEventBegin(SUMZ_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);
  MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

  // compute Z's locally

  ierr = VecSet(d->z, 0.0);
  CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(p, &low, &high);
  ierr = VecGetArray(p, &xp);
  ierr = VecGetArray(d->z, &zz);

  for (int i = d->firstContext; i <= d->lastContext; i++)
  {
    const int start = max(d->context[i], low) - low;
    const int end = min(d->context[i + 1], high) - low;
    double t = 0.0;
    for (int j = start; j < end; j++)
      t += xp[j];
    zz[i] = t;
  }

  PetscLogFlops((double)(high - low));

  ierr = VecRestoreArray(p, &xp);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(d->z, &zz);
  CHKERRQ(ierr);

  if (nprocs > 1)
  {

    // sum Z's globally

    VecSet(d->gz, 0.0);
    ierr = VecScatterBegin(d->sum_ctxt, d->z, d->gz, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(d->sum_ctxt, d->z, d->gz, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);

    // retrieve Z's locally

    // ierr = VecDestroy(d->z);CHKERRQ(ierr);
    // ierr = VecConvertMPIToSeqAll(d->gz,&d->z);CHKERRQ(ierr);

    ierr = VecScatterBegin(d->get_ctxt, d->gz, d->z, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(d->get_ctxt, d->gz, d->z, INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(SUMZ_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return 0;
}

// x = exp(x)

//#include "fpu_control.h"

#undef __FUNCT__
#define __FUNCT__ "VecExp"
int VecExp(Vec x)
{
  int n, ierr, overflow = 0;
  PetscScalar *xx;

  ierr = PetscLogEventBegin(VECEXP_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &n);
  CHKERRQ(ierr);
  ierr = VecGetArray(x, &xx);
  CHKERRQ(ierr);

  //fpu_control_t flags,oldflags;

  //_FPU_GETCW(flags);
  //_FPU_SETCW(flags|_FPU_MASK_UM|_FPU_MASK_OM|_FPU_MASK_PM|_FPU_MASK_DM);

  for (int i = 0; i < n; i++)
  {
    xx[i] = exp(xx[i]);
    //    if (isinf(xx[i]))
    //  overflow = 1;
  }

  //  _FPU_SETCW(flags);

  ierr = VecRestoreArray(x, &xx);
  CHKERRQ(ierr);
  ierr = PetscLogFlops((double)n);
  CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VECEXP_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return overflow;
}

// re-normalize a conditional probability distribution

#undef __FUNCT__
#define __FUNCT__ "normalize"
int normalize(Vec x, Dataset *d)
{
  PetscScalar *xx, *zz;
  int low, high, ierr;

  ierr = PetscLogEventBegin(NORMALIZE_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  // compute Z

  sum_z(x, d);
  VecPointwiseMult(d->z, d->pmarg, d->z);
  VecReciprocal(d->z);

  // x = x * z

  VecGetOwnershipRange(x, &low, &high);
  VecGetArray(x, &xx);
  VecGetArray(d->z, &zz);

  for (int i = d->firstContext; i <= d->lastContext; i++)
  {
    const int start = max(d->context[i], low) - low;
    const int end = min(d->context[i + 1], high) - low;
    for (int j = start; j < end; j++)
      xx[j] *= zz[i];
  }

  VecRestoreArray(x, &xx);
  VecRestoreArray(d->z, &zz);

  PetscLogFlops((double)(high - low));

  ierr = PetscLogEventEnd(NORMALIZE_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return 0;
}

// calculate log likelihood of a model

#undef __FUNCT__
#define __FUNCT__ "likelihood"
int likelihood(Vec x, double *f, Vec g, mle *e)
{
  int ierr, fail = 0;
  PetscScalar pen;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "likelihood options", NULL);

  // initialize

  ierr = PetscLogEventBegin(LIKELIHOOD_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  // compute probabilities given parameters

  if (e->d->penalty == L1 || e->d->penalty == BRIDGE)
  {

    double *xx;
    ierr = VecGetArray(x, &xx);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->p1, xx);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->p2, xx + e->d->nFeats);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &xx);
    CHKERRQ(ierr);

    VecCopy(e->p1, e->m->params);
    VecAXPY(e->m->params, -1.0, e->p2);
    MatMult(e->d->data, e->m->params, e->m->q);

    ierr = VecResetArray(e->p1);
    CHKERRQ(ierr);
    ierr = VecResetArray(e->p2);
    CHKERRQ(ierr);
  }
  else
  {
    MatMult(e->d->data, x, e->m->q);
  }

  fail = VecExp(e->m->q);

  PetscTruth memd;
  PetscOptionsName("-memd", "MEMD estimation", "Estimate", &memd);
  if (memd)
    VecPointwiseMult(e->m->q, e->d->p0, e->m->q);

  normalize(e->m->q, e->d);

  // likelihood

  //   f = - sum_i p_ref(x_i) log q(x_i)
  //   e->m->h = - sum_i q(x_i) log q(x_i)

  log_likelihood(e->d->p_ref, e->m->q, f, &(e->m->h));
  e->m->kl = *f - e->d->h0;

  if (e->d->penalty == L1)
  {

    // L1 (LASSO) penalty: f = f + sum(x) / lambda

    VecDot(x, e->d->sigma, &pen);
    //VecSum(x,&pen);
    //*f += pen/e->d->lambda;
    *f += pen;
  }
  else if (e->d->penalty == L2)
  {

    // L2 (ridge) penalty: f = f + sum(x^2) / 2*sigma

    const double half = 0.5;
    VecPointwiseMult(e->penalty, x, x);
    VecPointwiseMult(e->penalty, e->d->sigma, e->penalty);
    VecScale(e->penalty, half);
    VecSum(e->penalty, &pen);
    *f += pen;
  }

  // gradient Eq[f] - Ep_ref[f]

  if (e->d->penalty == L1)
  {

    // L1 penalty: grad = grad + 1/lambda

    double *gg;
    ierr = VecGetArray(g, &gg);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->g1, gg);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->g2, gg + e->d->nFeats);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(g, &gg);
    CHKERRQ(ierr);

    MatMultTransposeAdd(e->d->data, e->m->q, e->d->e_ref, e->g1);

    VecCopy(e->g1, e->g2);
    VecScale(e->g2, -1.0);

    ierr = VecResetArray(e->g1);
    CHKERRQ(ierr);
    ierr = VecResetArray(e->g2);
    CHKERRQ(ierr);

    //double const tmp = 1.0/e->d->lambda;
    //    VecShift(&tmp,g);
    VecAXPY(g, 1.0, e->d->sigma);
  }
  else
  {

    MatMultTransposeAdd(e->d->data, e->m->q, e->d->e_ref, g);

    if (e->d->penalty == L2)
    {

      // L2 penalty: grad = grad + x / sigma

      VecPointwiseMult(e->penalty, x, e->d->sigma);
      VecAXPY(g, 1.0, e->penalty);
    }
  }

  // if something went wrong, set f to NaN

  if (fail)
    SETERRQ(PETSC_ERR_FP, "Overflow");

  // finish up

  ierr = PetscLogEventEnd(LIKELIHOOD_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);
  e->fg++;

  PetscOptionsEnd();

  return 0;
}
