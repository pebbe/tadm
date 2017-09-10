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
// *** tadm
// ***

// $Id: tadm.cc,v 1.2 2005/09/26 19:40:58 jasonbaldridge Exp $

// Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"
#include "version.h"

#ifdef SVD
#include "svd.h"
#endif

#include <iostream>
#include <fstream>

int NORMALIZE_EVENT, LIKELIHOOD_EVENT, IIS_EVENT, UPDATE_EVENT,
    SUMZ_EVENT, ENTROPY_EVENT, LOAD_STAGE, INIT_STAGE, ESTIMATE_STAGE,
    FINAL_STAGE, VECEXP_EVENT, MEASURE_EVENT, SCAN_EVENT, READ_EVENT;

// register steepest ascent optimization method

EXTERN_C_BEGIN
extern int TaoCreate_STEEP(TAO_SOLVER);
EXTERN_C_END

// main entry point for driver

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  int ierr, nProcs, id;
  PetscLogDouble t0, ct0;
  PetscTruth expr = PETSC_FALSE, summary = PETSC_FALSE, iis = PETSC_FALSE,
             gis = PETSC_FALSE, ptron = PETSC_FALSE, valid = PETSC_FALSE;
  char now[LEN], filename[LEN], file_in[LEN];

  // initialize PETSc and TAO

  MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, (char *)0, "");
  TaoInitialize(&argc, &argv, (char *)0, "");

  MPI_Comm_size(PETSC_COMM_WORLD, &nProcs);
  MPI_Comm_rank(PETSC_COMM_WORLD, &id);

  // ignore signals (especially SIGCONT, to work well with condor scheduling)

  ierr = PetscPushSignalHandler(PETSC_NULL, PETSC_NULL);
  CHKERRQ(ierr);

  // check command line options

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Estimate Options",
                           "Estimate");
  CHKERRQ(ierr);

  PetscOptionsString("-events_in", "Event file to input", "Estimate", "", file_in, LEN, &valid);
  if (valid != PETSC_TRUE)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Input event file not specified");

  PetscOptionsName("-summary", "Print performance summary when done", "Estimate", &summary);
  PetscOptionsHasName(PETSC_NULL, "-expr", &expr);

  PetscOptionsGetString(PETSC_NULL, "-method", filename, LEN, &valid);
  if (valid == PETSC_TRUE)
  {
    PetscStrncmp(filename, "iis", 3, &iis);
    PetscStrncmp(filename, "gis", 3, &gis);
    PetscStrncmp(filename, "perceptron", 3, &ptron);
  }

  ierr = PetscOptionsEnd()
      CHKERRQ(ierr);

  // add steepest ascent method to TAO

  char path[] = "${TAO_DIR}/lib/lib${BOPT}/${PETSC_ARCH}/libtao.so";

  ierr = TaoRegisterDynamic("steep", path, "TaoCreate_STEEP", TaoCreate_STEEP);
  CHKERRQ(ierr);

  // set up logging

  PetscLogStageRegister(&LOAD_STAGE, "Load");
  PetscLogStageRegister(&INIT_STAGE, "Initialize");
  PetscLogStageRegister(&ESTIMATE_STAGE, "Estimate");
  PetscLogStageRegister(&FINAL_STAGE, "Finalize");

  PetscLogEventRegister(&NORMALIZE_EVENT, "Normalize", 0);
  PetscLogEventRegister(&LIKELIHOOD_EVENT, "Log-likelihood", 0);
  PetscLogEventRegister(&IIS_EVENT, "Iterative Scaling", 0);
  PetscLogEventRegister(&SUMZ_EVENT, "Sum Z", 0);
  PetscLogEventRegister(&ENTROPY_EVENT, "Entropy", 0);
  PetscLogEventRegister(&VECEXP_EVENT, "VecExp", 0);
  PetscLogEventRegister(&MEASURE_EVENT, "Measure file", 0);
  PetscLogEventRegister(&SCAN_EVENT, "Scan file", 0);
  PetscLogEventRegister(&READ_EVENT, "Read file", 0);

  // write log header

  PetscPrintf(PETSC_COMM_WORLD, "\nMaximum Entropy Parameter Estimation\n");
  PetscPrintf(PETSC_COMM_WORLD, "  %s\n\n", VERSION);

  PetscGetDate(now, LEN);
  PetscPrintf(PETSC_COMM_WORLD, "Start: %s\n\n", now);

  // initialize event space

  PetscLogStagePush(LOAD_STAGE);

  Dataset training;

  PetscPrintf(PETSC_COMM_WORLD, "Events in  = %s\n", file_in);
  training.readEvents(file_in);

  PetscLogStagePop();

#ifdef SVD
  // reduce

  Mat U, V;
  int nev;

  PetscOptionsGetInt(PETSC_NULL, "-svd", &nev, &valid);

  if (valid)
  {
    svd(training.data, &nev, &U, &V);
    training.replaceEvents(U, V);
  }
#endif /* SVD */

  PetscLogStagePush(INIT_STAGE);
  initializeDataset(training, iis);
  PetscLogStagePop();

  if (summary)
  {
    PetscGetTime(&t0);
    PetscGetCPUTime(&ct0);
  }

  // initialize model

  Model fitted(&training);

  // write info about empirical distribution

  PetscOptionsHasName(PETSC_NULL, "-uniform", &valid);
  if (valid)
    PetscPrintf(PETSC_COMM_WORLD, "Marginal   = uniform\n");
  else
    PetscPrintf(PETSC_COMM_WORLD, "Marginal   = pseudo-likelihood\n");
  PetscPrintf(PETSC_COMM_WORLD, "Smoothing  = none\n");
  PetscPrintf(PETSC_COMM_WORLD, "Procs      = %d\n", nProcs);

  PetscPrintf(PETSC_COMM_WORLD, "\nClasses    = %d\n", training.nClasses);
  PetscPrintf(PETSC_COMM_WORLD, "Contexts   = %d\n", training.nContexts);
  PetscPrintf(PETSC_COMM_WORLD, "Features   = %d / %d\n", training.nActive,
              training.nFeats);
  PetscPrintf(PETSC_COMM_WORLD, "Non-zeros  = %d\n", training.nNZeros);

  PetscPrintf(PETSC_COMM_WORLD, "\nH(p_ref)   = %14.8e\n", training.h0);

  // do it

  mle estimator(&fitted);

  if (training.bootstrap)
  {

    for (int i = 0; i < training.bootstrap; i++)
    {

      initializeDataset(training, iis);

      PetscLogStagePush(ESTIMATE_STAGE);

      if (iis)
        std::cerr << "Arrrggghhhhh!" << std::endl;
      else if (gis)
        std::cerr << "Arrrggghhhhh!" << std::endl;
      else
        estimate_params(&estimator);

      PetscLogStagePop();

      // write results

      PetscLogStagePush(FINAL_STAGE);
      writeParams(&fitted, i);
      PetscLogStagePop();

      // reinitialize model

      estimator.fg = 0;
      estimator.its = 0;
      VecSet(fitted.params, 0.0);
      VecSet(fitted.q, 0.0);
    }
  }
  else
  {

    PetscLogStagePush(ESTIMATE_STAGE);

    if (iis)
      estimate_params_iis(&estimator);
    else if (gis)
      estimate_params_gis(&estimator);
    else if (ptron)
      estimate_params_perceptron(&estimator);
    else
      estimate_params(&estimator);

    PetscLogStagePop();

    // write results

    PetscLogStagePush(FINAL_STAGE);
    writeParams(&fitted, -1);
    PetscLogStagePop();
  }

  if (summary)
  {

    // print performance summary

    PetscLogDouble t1, ct1, flops, tflops, smax;

    PetscGetTime(&t1);
    PetscGetCPUTime(&ct1);
    PetscGetFlops(&flops);
    PetscMallocGetMaximumUsage(&smax);

    MPI_Reduce(&flops, &tflops, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

    PetscPrintf(PETSC_COMM_WORLD, "\nMethod       : %s\n", estimator.method);
    PetscPrintf(PETSC_COMM_WORLD, "Events in    : %s\n", file_in);
    PetscPrintf(PETSC_COMM_WORLD, "Final KL     : %e nats\n", fitted.kl);
    PetscPrintf(PETSC_COMM_WORLD, "Iterations   : %d\n", estimator.its);
    PetscPrintf(PETSC_COMM_WORLD, "LL evals     : %d\n", estimator.fg);
    PetscPrintf(PETSC_COMM_WORLD, "Elapsed time : %.2f secs\n", t1 - t0);
    PetscPrintf(PETSC_COMM_WORLD, "Throughput   : %.2f MFLOPS\n",
                (tflops / 1e6) / (t1 - t0));

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                            "CPU time     : %.2f secs [CPU %d]\n",
                            ct1 - ct0, id);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                            "Memory used  : %.2f Mbytes [CPU %d]\n",
                            smax / (1024.0 * 1024.0), id);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);

    // Give a one-line summary (easy to extract from logs)

    if (expr)
      PetscPrintf(PETSC_COMM_WORLD, "\nEXPR: %s %s %e %d %d %.2f %.2f\n",
                  estimator.method, file_in, fitted.kl, estimator.its,
                  estimator.fg, t1 - t0, (tflops / 1e6) / (t1 - t0));
  }

  PetscGetDate(now, LEN);
  PetscPrintf(PETSC_COMM_WORLD, "\nEnd: %s\n\n", now);

  //  Dump unused options and quit

  PetscOptionsLeft();

  ierr = TaoFinalize();
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  CHKERRQ(ierr);
  MPI_Finalize();
  return 0;
}
