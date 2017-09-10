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

// $Id: steep.cc,v 1.2 2007/09/03 19:16:48 jasonbaldridge Exp $

#include "src/tao_impl.h"

typedef struct {

  TaoVec *DX;
  TaoVec *WW;

} TAO_STEEP;


static int TaoView_STEEP(TAO_SOLVER tao,void* solver);
static int TaoSolve_STEEP(TAO_SOLVER tao,void *solver);
static int TaoSetOptions_STEEP(TAO_SOLVER tao, void *solver);
static int TaoSetUp_STEEP(TAO_SOLVER tao, void *solver);

/*
   Implements gradient descent method with a line search 
   for solving unconstrained minimization problems.
*/

#undef __FUNCT__  
#define __FUNCT__ "TaoSolve_STEEP"
static int TaoSolve_STEEP(TAO_SOLVER tao,void *solver)
{
  TAO_STEEP       *gradP = (TAO_STEEP *) solver;
  int                iter=0, info, line=0;
  double             f, gnorm, step=0.0, gdx, f_full;
  TaoVec             *gg, *xx;
  TaoVec             *dx=gradP->DX, *ww=gradP->WW;
  TaoTerminateReason reason;

  TaoFunctionBegin;

  info=TaoGetSolution(tao,&xx);CHKERRQ(info);
  info=TaoGetGradient(tao,&gg);CHKERRQ(info);

  info = TaoComputeMeritFunctionGradient(tao,xx,&f,gg);CHKERRQ(info);
  info = gg->Norm2(&gnorm);CHKERRQ(info);         /* gnorm = || gg || */

  while (1) {

    info = TaoMonitor(tao,iter++,f,gnorm,0.0,step,&reason);CHKERRQ(info);
    if (reason!=TAO_CONTINUE_ITERATING) break;

    /* Descent direction */
    info = dx->ScaleCopyFrom(-1.0,gg); CHKERRQ(info);
    //    info = dx->Dot(gg,&gdx); CHKERRQ(info);

    /* Line search */
    step=1.0;
    info = TaoLineSearchApply(tao,xx,gg,dx,ww,&f,&f_full,&step,&line);
    CHKERRQ(info);
    info = gg->Norm2(&gnorm);CHKERRQ(info);

  }

  TaoFunctionReturn(0);
}
/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSetUp_STEEP"
static int TaoSetUp_STEEP(TAO_SOLVER tao, void *solver)
{
  int        info;
  TAO_STEEP *ctx = (TAO_STEEP *)solver;
  TaoVec *xx;

  TaoFunctionBegin;
  info = TaoCheckFG(tao);CHKERRQ(info);
  info = TaoGetSolution(tao,&xx);CHKERRQ(info);

  info=xx->Clone(&ctx->DX);CHKERRQ(info);
  info=xx->Clone(&ctx->WW);CHKERRQ(info);

  info = TaoLineSearchSetUp(tao);CHKERRQ(info);

  TaoFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoDestroy_STEEP"
static int TaoDestroy_STEEP(TAO_SOLVER tao, void *solver)
{
  TAO_STEEP *ctx = (TAO_STEEP *)solver;
  int     info;

  TaoFunctionBegin;

  if (tao->setupcalled) {
    if (ctx->DX){ info=TaoVecDestroy(ctx->DX);CHKERRQ(info); }
    if (ctx->WW){ info=TaoVecDestroy(ctx->WW);CHKERRQ(info); }
  }
  info = TaoLineSearchDestroy(tao);CHKERRQ(info);

  TaoFree(ctx);
  TaoFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSetOptions_STEEP"
static int TaoSetOptions_STEEP(TAO_SOLVER tao, void *solver)
{
  int        info;

  TaoFunctionBegin;
  info = TaoOptionsHead("Gradient descent method for unconstrained optimization");CHKERRQ(info);

  info = TaoLineSearchSetFromOptions(tao);CHKERRQ(info);

  info = TaoOptionsTail();CHKERRQ(info);

  TaoFunctionReturn(0);
}


/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoView_STEEP"
static int TaoView_STEEP(TAO_SOLVER tao,void* solver)
{
  int     info;

  TaoFunctionBegin;

  info = TaoLineSearchView(tao);CHKERRQ(info);

  TaoFunctionReturn(0);
}

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoCreate_STEEP"
int TaoCreate_STEEP(TAO_SOLVER tao)
{
  TAO_STEEP    *gradP;
  int             info;

  TaoFunctionBegin;

  info = TaoNew(TAO_STEEP,&gradP); CHKERRQ(info);
  PetscLogObjectMemory(tao,sizeof(TAO_STEEP));

  info = TaoSetTaoSolveRoutine(tao,TaoSolve_STEEP,(void*)gradP); 
  CHKERRQ(info);
  info = TaoSetTaoSetUpDownRoutines(tao,TaoSetUp_STEEP,TaoDestroy_STEEP);
  CHKERRQ(info);
  info = TaoSetTaoOptionsRoutine(tao,TaoSetOptions_STEEP);
  CHKERRQ(info);
  info=TaoSetTaoViewRoutine(tao,TaoView_STEEP); 
  CHKERRQ(info);

  info = TaoSetMaximumIterates(tao,50); CHKERRQ(info);
  info = TaoSetTolerances(tao,1e-16,1e-16,0,0); CHKERRQ(info);

  info = TaoCreateMoreThuenteLineSearch(tao,0,0); CHKERRQ(info);

  TaoFunctionReturn(0);
}
EXTERN_C_END

