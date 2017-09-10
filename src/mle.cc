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

/// ***
/// *** mle
/// ***

// $Id: mle.cc,v 1.4 2006/03/17 01:36:31 malouf Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"

#include <iostream>

// constructor

mle::mle(Model *model)
{
  PetscTruth valid;

  // set up

  m = model;
  d = m->d;
  fg = 0;
  its = 0;

  // extract options

  PetscMalloc(LEN * sizeof(char), &method);
  PetscOptionsGetString(PETSC_NULL, "-method", method, LEN, &valid);
  if (valid != PETSC_TRUE)
    if (d->penalty == L1 || d->penalty == BRIDGE)
      PetscStrcpy(method, "tao_blmvm");
    else
      PetscStrcpy(method, "tao_lmvm");

  PetscOptionsHasName(PETSC_NULL, "-monitor", &monitor);

  PetscOptionsGetInt(PETSC_NULL, "-checkpoint", &checkpoint, &valid);
  if (valid == PETSC_TRUE)
  {
    monitor = PETSC_TRUE;
  }
  else
  {
    checkpoint = 0;
  }

  PetscOptionsGetInt(PETSC_NULL, "-max_it", &max_it, &valid);
  if (valid != PETSC_TRUE)
    max_it = 9999;

  PetscOptionsGetReal(PETSC_NULL, "-frtol", &frtol, &valid);
  if (valid != PETSC_TRUE)
    frtol = 1e-8;

  PetscOptionsGetReal(PETSC_NULL, "-fatol", &fatol, &valid);
  if (valid != PETSC_TRUE)
    fatol = 1e-10;

  lastf = 0.0;
}

// simple convergence test

int tao_maxent_conv(TAO_SOLVER tao, void *e)
{
  int i;
  double f, norm, cnorm, xdiff;
  TaoTerminateReason reason;

  TaoGetIterationData(tao, &i, &f, &norm, &cnorm, &xdiff, &reason);
  reason = maxent_conv(i, f, (mle *)e);
  TaoSetTerminationReason(tao, reason);

  return 0;
}

TaoTerminateReason
maxent_conv(int i, double f, mle *e)
{
  double fr, fa;

  // check for convergence

  fr = fabs(f - e->lastf) / (f + 1e-15);
  fa = fabs(f - e->lastf);
  e->lastf = f;

  if (i >= e->max_it)
  {
    return TAO_DIVERGED_MAXITS;
  }
  else if (fr <= e->frtol)
  {
    return TAO_CONVERGED_RTOL;
  }
  else if (fa <= e->fatol)
  {
    return TAO_CONVERGED_ATOL;
  }
  else
  {
    return (TaoTerminateReason)0;
  }
}

#undef __FUNCT__
#define __FUNCT__ "maxent_monitor"
int maxent_monitor(TAO_SOLVER tao, void *e)
{
  int i;
  double f, norm, cnorm, xdiff;
  TaoTerminateReason reason;

  // display iteration statistics

  TaoGetIterationData(tao, &i, &f, &norm, &cnorm, &xdiff, &reason);
  PetscPrintf(PETSC_COMM_WORLD, "%4d%18.8e%18.8e%18.8e\n",
              i, ((mle *)e)->m->kl, ((mle *)e)->m->h, norm);

  ((mle *)e)->its = i;

  // dump parameters

  if ((((mle *)e)->checkpoint != 0) && (i % ((mle *)e)->checkpoint == 0))
    writeParams(((mle *)e)->m, i);

  // if stopping, explain

  if (reason != 0)
  {
    switch (reason)
    {
    case TAO_CONVERGED_ATOL:
      PetscPrintf(PETSC_COMM_WORLD, "\nConverged: res <= atol\n");
      break;
    case TAO_CONVERGED_RTOL:
      PetscPrintf(PETSC_COMM_WORLD, "\nConverged: res/res0 <= rtol\n");
      break;
    case TAO_CONVERGED_TRTOL:
      PetscPrintf(PETSC_COMM_WORLD, "\nConverged: xdiff <= trtol\n");
      break;
    case TAO_CONVERGED_MINF:
      PetscPrintf(PETSC_COMM_WORLD, "\nConverged: f <= fmin\n");
      break;
    case TAO_DIVERGED_MAXITS:
      PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: its>maxits\n");
      break;
    case TAO_DIVERGED_NAN:
      PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: Numerical problems\n");
      break;
    case TAO_DIVERGED_MAXFCN:
      PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: nfunc > maxnfuncts\n");
      break;
    case TAO_DIVERGED_LS_FAILURE:
      PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: line search failure\n");
      break;
    case TAO_DIVERGED_TR_REDUCTION:
      PetscPrintf(PETSC_COMM_WORLD, "\nDiverged: TR reduction\n");
      break;
    default:
      PetscPrintf(PETSC_COMM_WORLD, "\nHelp me!\n");
    }
  }

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "tao_likelihood"
int tao_likelihood(TAO_SOLVER tao, Vec x, double *f, Vec g, void *ctx)
{
  likelihood(x, f, g, (mle *)ctx);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "tao_gradient"
int tao_gradient(TAO_SOLVER tao, Vec x, Vec g, void *ctx)
{
  double f;

  likelihood(x, &f, g, (mle *)ctx);

  return 0;
}

typedef struct
{
  int (*ggg)(TAO_SOLVER, Vec, Vec, void *);
  TAO_SOLVER tao;
  void *userctx;
} PetscFDAppCtx;

#if 0
#undef __FUNCT__
#define __FUNCT__ "Ftemp"
static int 
Ftemp(SNES snes ,Vec X,Vec G,void*ctx)
{
  int ierr;
  PetscFDAppCtx* pctx=( PetscFDAppCtx*) ctx;

  ierr = (*pctx->ggg)(pctx->tao,X,G,pctx->userctx); CHKERRQ(ierr);

  return 0;
}
#endif

// maximum likelihood estimation

#undef __FUNCT__
#define __FUNCT__ "estimate_params"
int estimate_params(mle *e)
{
  int ierr;
  PetscTruth uvalid, lvalid, valid;
  double ubound, lbound;

  // create solver

  ierr = TaoCreate(PETSC_COMM_WORLD, e->method, &e->tao);
  CHKERRQ(ierr);
  ierr = TaoApplicationCreate(PETSC_COMM_WORLD, &e->tao_appl);
  CHKERRQ(ierr);
  ierr = TaoSetObjectiveAndGradientRoutine(e->tao, tao_likelihood, e);
  CHKERRQ(ierr);

  // check for bounds on parameters

  PetscOptionsGetReal(PETSC_NULL, "-ubound", &ubound, &uvalid);
  PetscOptionsGetReal(PETSC_NULL, "-lbound", &lbound, &lvalid);

  // do it

  if (e->d->penalty == L1 || e->d->penalty == BRIDGE)
  {

    int nprocs;

    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    if (nprocs > 1)
    {
      SETERRQ(PETSC_ERR_SUP, "Multiprocessor L1 penalty not implemented yet");
      e->d->penalty = NONE;
    }
    else
    {

      // initialize parameter vectors and gradient

      int const n = e->d->nFeats;

      ierr = VecCreateSeq(PETSC_COMM_WORLD, n * 2, &e->params2);
      CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_WORLD, n, PETSC_NULL, &e->p1);
      CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_WORLD, n, PETSC_NULL, &e->p2);
      CHKERRQ(ierr);
      ierr = VecSet(e->params2, 0.0);
      CHKERRQ(ierr);

      //ierr = VecCreateSeq(PETSC_COMM_WORLD,n*2,&e->g); CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_WORLD, n, PETSC_NULL, &e->g1);
      CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_WORLD, n, PETSC_NULL, &e->g2);
      CHKERRQ(ierr);

      if (e->d->penalty == BRIDGE)
        ierr = VecCreateSeq(PETSC_COMM_WORLD, 2 * n, &e->penalty);
      CHKERRQ(ierr);

      ierr = TaoSetInitialVector(e->tao, e->params2);
      CHKERRQ(ierr);

      Vec upper, lower;
      ierr = VecDuplicate(e->params2, &upper);
      CHKERRQ(ierr);
      ierr = VecDuplicate(e->params2, &lower);
      CHKERRQ(ierr);
      VecSet(lower, 0.0);
      VecSet(upper, TAO_INFINITY);
      TaoSetVariableBounds(e->tao, lower, upper);
      //    ierr = VecDestroy(lower); CHKERRQ(ierr);
      //    ierr = VecDestroy(upper); CHKERRQ(ierr);
    }
  }
  else
  {

    if (e->d->penalty)
      ierr = VecDuplicate(e->m->params, &e->penalty);
    CHKERRQ(ierr);

    ierr = TaoSetInitialVector(e->tao, e->m->params);
    CHKERRQ(ierr);
  }

  // set up bounds

  if (uvalid == PETSC_TRUE || lvalid == PETSC_TRUE)
  {

    if (e->d->penalty == L1 || e->d->penalty == BRIDGE)
    {

      SETERRQ(PETSC_ERR_SUP, "Bound contraints not implemented for L1 penalty yet");
    }
    else
    {

      Vec upper, lower;

      ierr = VecDuplicate(e->m->params, &upper);
      CHKERRQ(ierr);
      ierr = VecDuplicate(e->m->params, &lower);
      CHKERRQ(ierr);
      if (lvalid == PETSC_TRUE)
        VecSet(lower, lbound);
      else
        VecSet(lower, TAO_NINFINITY);
      if (uvalid == PETSC_TRUE)
        VecSet(upper, ubound);
      else
        VecSet(upper, TAO_INFINITY);
        TaoSetVariableBounds(e->tao, lower, upper);
    }
  }

  // set up convergence conditions

  PetscOptionsHasName(PETSC_NULL, "-converge", &valid);
  if (valid)
  {
    e->lastf = 0.0;
    TaoSetConvergenceTest(e->tao, tao_maxent_conv, e);
  }
  else
  {
    TaoSetMaximumIterations(e->tao, e->max_it);
    TaoSetTolerances(e->tao, e->fatol, e->frtol, 0.0, 0.0);
  }

  // start logging

  if (e->monitor)
  {
    PetscPrintf(PETSC_COMM_WORLD, "\n It       D(p_ref||p)              H(p)              Norm\n\n");
    TaoSetMonitor(e->tao, maxent_monitor, e);
  }

  // do it

  // ierr = TaoLMVMSetSize(e->tao,2); CHKERRQ(ierr);
  /* TODO
  ierr = TaoSetApplication(e->tao, e->tao_appl);
  CHKERRQ(ierr);
  */
  ierr = TaoSetFromOptions(e->tao);
  CHKERRQ(ierr);
  ierr = TaoSolve(e->tao);
  CHKERRQ(ierr);

  // get final answer

  if (e->d->penalty == L1)
  {
    double *xx;

    ierr = VecGetArray(e->params2, &xx);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->p1, xx);
    CHKERRQ(ierr);
    ierr = VecPlaceArray(e->p2, xx + e->d->nFeats);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(e->params2, &xx);
    CHKERRQ(ierr);

    VecCopy(e->p1, e->m->params);
    VecAXPY(e->m->params, -1.0, e->p2);
  }

  // clean up

  if (e->d->penalty == L1)
  {
    ierr = VecDestroy(e->params2);
    CHKERRQ(ierr);
    ierr = VecDestroy(e->p1);
    CHKERRQ(ierr);
    ierr = VecDestroy(e->p2);
    CHKERRQ(ierr);
    ierr = VecDestroy(e->g1);
    CHKERRQ(ierr);
    ierr = VecDestroy(e->g2);
    CHKERRQ(ierr);
  }
  else if (e->d->penalty != NONE)
    ierr = VecDestroy(e->penalty);
  CHKERRQ(ierr);

  ierr = TaoAppDestroy(e->tao_appl);
  CHKERRQ(ierr);
  ierr = TaoDestroy(e->tao);
  CHKERRQ(ierr);

  return 0;
}
