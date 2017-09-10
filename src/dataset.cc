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
// *** Dataset
// ***

// $Id: dataset.cc,v 1.3 2006/03/17 01:36:31 malouf Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"
#include "fileio.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

// constructor

Dataset::Dataset()
{
  PetscTruth valid;

  // extract options

  PetscOptionsHasName(PETSC_NULL, "-correction", &valid);
  correct = valid;

  PetscOptionsGetInt(PETSC_NULL, "-bootstrap", &bootstrap, &valid);
  if (!valid)
    bootstrap = 0;

  initialized = false;
}

// decompose 1D problem (stolen from MPE)

#undef __FUNCT__
#define __FUNCT__ "Decomp1d"
int Decomp1d(int n, int size, int rank, int *s, int *e)
{
  int nlocal, deficit;

  nlocal = n / size;
  *s = rank * nlocal + 1;
  deficit = n % size;
  *s = *s + ((rank < deficit) ? rank : deficit);
  if (rank < deficit)
    nlocal++;
  *e = *s + nlocal - 1;
  if (*e > n || rank == size - 1)
    *e = n;
  return MPI_SUCCESS;
}

// scan file to measure its size

#undef __FUNCT__
#define __FUNCT__ "measureFile"
int measureFile(Datafile &f, int *nClasses, int *nContexts, int *nFeats,
                int *nNZeros, double *c)
{
  double v, vv;
  int ii, jj, k, id, ierr;

  ierr = PetscLogEventBegin(MEASURE_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  MPI_Comm_rank(PETSC_COMM_WORLD, &id);

  if (id == 0)
  {

    *nClasses = *nContexts = *nFeats = *nNZeros = 0;
    *c = 0.0;
    f.firstContext();

    PetscTruth memd;
    PetscOptionsName("-memd", "MEMD estimation", "Estimate", &memd);

    // read file

    while (f.getCount(&ii) != EOF)
    {
      if (ii > 0)
      {
        (*nContexts)++;
        *nClasses += ii;
        ;
        for (int i = 0; i < ii; i++)
        {
          int istat;
          if (memd)
            istat = f.getFreq(&v, &v, &jj);
          else
            istat = f.getFreq(&v, &jj);
          if (istat == EOF)
            SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
          vv = 0.0;
          for (int j = 0; j < jj; j++)
          {
            if (f.getPair(&k, &v) == EOF)
              SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
            vv += v;
            *nFeats = max(k, *nFeats);
          }
          *c = max(vv, *c);
          *nNZeros += jj;
        }
      }
    }

    (*nFeats)++; // features are numbered starting from zero
  }

  MPI_Bcast(nClasses, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(nContexts, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(nFeats, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(nNZeros, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(c, 1, MPI_INT, 0, PETSC_COMM_WORLD);

  ierr = PetscLogEventEnd(MEASURE_EVENT, 0, 0, 0, 0);
  CHKERRQ(ierr);

  return 0;
}

// read events file

#undef __FUNCT__
#define __FUNCT__ "Dataset::readEvents"
int Dataset::readEvents(char *filename)
{
  int id, nProcs, ierr;
  double freq, prior, vv;
  PetscTruth valid;

  MPI_Comm_size(PETSC_COMM_WORLD, &nProcs);
  MPI_Comm_rank(PETSC_COMM_WORLD, &id);

  PetscOptionsHasName(PETSC_NULL, "-load", &valid);
  if (valid)
  {

#if 0

    // quick load binary data file

    MatInfo info;
    char tmp[LEN];

    ierr = PetscLogEventBegin(READ_EVENT,0,0,0,0);CHKERRQ(ierr);

    // get event probabilities

    sprintf(tmp,"%s.p_ref.gz",filename);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,tmp,PETSC_BINARY_RDONLY,&v);
    CHKERRQ(ierr);
    if (nProcs > 1)
      VecLoad(v,VECMPI,&p_ref);
    else
      VecLoad(v,VECSEQ,&p_ref);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(v);CHKERRQ(ierr);

    // get feature matrix

    sprintf(tmp,"%s.data.gz",filename);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,tmp,PETSC_BINARY_RDONLY,&v);
    CHKERRQ(ierr);
    if (nProcs > 1)
      MatLoad(v,MATMPIAIJ,&data);
    else
      MatLoad(v,MATSEQAIJ,&data);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(v);CHKERRQ(ierr);

    // get dataset size

    MatGetSize(data,nClasses,nFeats);
    MatGetInfo(data,MAT_GLOBAL_SUM,&info);
    nClasses = int(info.rows_global);
    nFeats = int(info.columns_global);
    nNZeros = int(info.nz_used);

    // set up e_ref
      
    if (nProcs > 1)

    else
      ierr = VecCreateSeq(PETSC_COMM_SELF,nFeats,&e_ref);CHKERRQ(ierr);

    ierr = VecSet(e_ref,0.0);CHKERRQ(ierr);
      
    // set up z
    
    ierr = VecCreateSeq(PETSC_COMM_SELF,nContexts,&z);CHKERRQ(ierr);

    ierr = PetscLogEventEnd(READ_EVENT,0,0,0,0);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP, "Binary event file format not implemented");
#endif
  }
  else
  {

    // open input file

    Datafile in(filename);

    measureFile(in, &nClasses, &nContexts, &nFeats, &nNZeros, &c);

    // add a correction feature, if necessary
    if (correct)
    {
      nFeats++;
      nNZeros += nClasses;
    }

    // allocate storage

    ierr = PetscMalloc((nContexts + 1) * sizeof(int), &context);
    CHKERRQ(ierr);

    // read in file

    PetscTruth memd;
    PetscOptionsName("-memd", "MEMD estimation", "Estimate", &memd);

    if (nProcs == 1)
    {

      // uniprocessor version

      int *ptr, *row, iPoint, iClass, iContext, ii, jj;
      double *val;

      // allocate storage for events

      ierr = PetscMalloc((nClasses + 1) * sizeof(int), &ptr);
      CHKERRQ(ierr);
      ierr = PetscMalloc(nNZeros * sizeof(int), &row);
      CHKERRQ(ierr);
      ierr = PetscMalloc(nNZeros * sizeof(double), &val);
      CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, nClasses, &p_ref);
      CHKERRQ(ierr);
      if (memd)
        ierr = VecCreateSeq(PETSC_COMM_SELF, nClasses, &p0);
      CHKERRQ(ierr);

      // rewind to first context

      in.firstContext();

      // read events as sparse matrix

      iPoint = iClass = iContext = 0;

      ierr = PetscLogEventBegin(READ_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      while (in.getCount(&ii) != EOF)
      {
        if (ii > 0)
        {
          context[iContext] = iClass;
          iContext++;
          for (int i = 0; i < ii; i++)
          {
            int istat;
            if (memd)
              istat = in.getFreq(&freq, &prior, &jj);
            else
              istat = in.getFreq(&freq, &jj);
            if (istat == EOF)
              SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
            ierr = VecSetValue(p_ref, iClass, freq, INSERT_VALUES);
            if (memd)
              ierr = VecSetValue(p0, iClass, prior, INSERT_VALUES);
            ptr[iClass] = iPoint;
            vv = 0.0;
            for (int j = 0; j < jj; j++)
            {
              if (in.getPair(&row[iPoint], &val[iPoint]) == EOF)
                SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
              vv += val[iPoint];
              iPoint++;
            }
            // add correction feature
            if (correct)
            {
              row[iPoint] = nFeats - 1;
              val[iPoint] = c - vv;
              iPoint++;
            }
            iClass++;
          }
        }
      }

      ptr[iClass] = iPoint;
      context[iContext] = iClass;
      firstContext = 0;
      lastContext = nContexts - 1;

      ierr = PetscLogEventEnd(READ_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      // convert to PETSc sparse matrix

      ierr = VecAssemblyBegin(p_ref);
      CHKERRQ(ierr);
      if (memd)
        ierr = VecAssemblyBegin(p0);
      CHKERRQ(ierr);
      ierr = VecAssemblyEnd(p_ref);
      CHKERRQ(ierr);
      if (memd)
        ierr = VecAssemblyEnd(p0);
      CHKERRQ(ierr);

      ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nClasses, nFeats,
                                       ptr, row, val, &data);
      CHKERRQ(ierr);

      // set up e_ref

      ierr = VecCreateSeq(PETSC_COMM_SELF, nFeats, &e_ref);
      CHKERRQ(ierr);
      ierr = VecSet(e_ref, 0.0);
      CHKERRQ(ierr);

#if SCALED
      ierr = VecCreateSeq(PETSC_COMM_SELF, nFeats, &v_ref);
      CHKERRQ(ierr);
      ierr = VecSet(v_ref, 0.0);
      CHKERRQ(ierr);
#endif

      // set up z

      ierr = VecCreateSeq(PETSC_COMM_SELF, nContexts, &z);
      CHKERRQ(ierr);
    }
    else
    {

      // multiprocessor version

      int e0, e1, f0, f1, *nnz, *dnz, iClass, iContext, k, ii, jj;
      double v;

      if (memd)
        SETERRQ(PETSC_ERR_SUP, "Multiprocessor MEMD estimation not implemented yet");

      // partition event set

      Decomp1d(nClasses, nProcs, id, &e0, &e1);
      Decomp1d(nFeats, nProcs, id, &f0, &f1);

      e0--;
      e1--;
      f0--;
      f1--;

      ierr = PetscMalloc((e1 - e0 + 1) * sizeof(int), &dnz);
      CHKERRQ(ierr);
      ierr = PetscMalloc((e1 - e0 + 1) * sizeof(int), &nnz);
      CHKERRQ(ierr);

      // scan file to count non-zeros

      in.firstContext();

      iClass = iContext = 0;
      firstContext = -1;

      ierr = PetscLogEventBegin(SCAN_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      while ((in.getCount(&ii) != EOF) && (iClass <= e1))
      {
        if (ii > 0)
        {
          context[iContext] = iClass;
          for (int i = 0; i < ii; i++)
          {
            if ((iClass >= e0) && (iClass <= e1))
            {
              if (firstContext == -1)
              {
                firstContext = iContext;
              }
              lastContext = iContext;
              // read in line
              if (in.getFreq(&freq, &jj) == EOF)
                SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
              dnz[iClass - e0] = nnz[iClass - e0] = 0;
              for (int j = 0; j < jj; j++)
              {
                if (in.getPair(&k, &freq) == EOF)
                  SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
                if ((k >= f0) && (k <= f1))
                {
                  dnz[iClass - e0]++;
                }
                else
                {
                  nnz[iClass - e0]++;
                }
              }
              // add correction feature
              if (correct)
              {
                if ((nFeats - 1 >= f0) && (nFeats - 1 <= f1))
                {
                  dnz[iClass - e0]++;
                }
                else
                {
                  nnz[iClass - e0]++;
                }
              }
            }
            else
            {
              // skip to next line
              in.skipLine();
            }
            iClass++;
          }
          iContext++;
        }
      }

      context[iContext] = iClass;

      ierr = PetscLogEventEnd(SCAN_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      // allocate storage for events

      ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, (e1 - e0 + 1), (f1 - f0 + 1), nClasses,
                             nFeats, 0, dnz, 0, nnz, &data);
      CHKERRQ(ierr);

      ierr = PetscFree(dnz);
      CHKERRQ(ierr);
      ierr = PetscFree(nnz);
      CHKERRQ(ierr);

      ierr = VecCreateMPI(PETSC_COMM_WORLD, (e1 - e0 + 1), nClasses, &p_ref);
      CHKERRQ(ierr);

      // read events as parallel sparse matrix

      in.firstContext();

      iClass = 0;

      ierr = PetscLogEventBegin(READ_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      while ((in.getCount(&ii) != EOF) && (iClass <= e1))
      {
        if (ii > 0)
        {
          for (int i = 0; i < ii; i++)
          {
            if ((iClass >= e0) && (iClass <= e1))
            {
              // read in line
              if (in.getFreq(&freq, &jj) == EOF)
                SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
              ierr = VecSetValue(p_ref, iClass, freq, INSERT_VALUES);
              vv = 0.0;
              for (int j = 0; j < jj; j++)
              {
                if (in.getPair(&k, &v) == EOF)
                  SETERRQ(PETSC_ERR_FILE_READ, "Error reading data file");
                MatSetValues(data, 1, &iClass, 1, &k, &v, INSERT_VALUES);
                vv += v;
              }
              // add correction feature
              if (correct)
              {
                k = nFeats - 1;
                v = c - vv;
                MatSetValues(data, 1, &iClass, 1, &k, &v, INSERT_VALUES);
              }
            }
            else
            {
              // skip to next line
              in.skipLine();
            }
            iClass++;
          }
        }
      }

      ierr = PetscLogEventEnd(READ_EVENT, 0, 0, 0, 0);
      CHKERRQ(ierr);

      // assemble data structures

      ierr = VecAssemblyBegin(p_ref);
      CHKERRQ(ierr);
      ierr = MatAssemblyBegin(data, MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);
      ierr = VecAssemblyEnd(p_ref);
      CHKERRQ(ierr);
      ierr = MatAssemblyEnd(data, MAT_FINAL_ASSEMBLY);
      CHKERRQ(ierr);

      // set up e_ref

      ierr = VecCreateMPI(PETSC_COMM_WORLD, (f1 - f0 + 1), nFeats, &e_ref);
      CHKERRQ(ierr);

#if SCALED
      ierr = VecCreateMPI(PETSC_COMM_WORLD, (f1 - f0 + 1), nFeats, &v_ref);
      CHKERRQ(ierr);
      ierr = VecSet(v_ref, 0.0);
      CHKERRQ(ierr);
#endif

      // set up z, gz

      ierr = VecCreateSeq(PETSC_COMM_SELF, nContexts, &z);
      CHKERRQ(ierr);
      ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, nContexts, &gz);
      CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD, lastContext - firstContext + 1,
                            firstContext, 1, &is_z);
      CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_WORLD, lastContext - firstContext + 1,
                            firstContext, 1, &is_gz);
      CHKERRQ(ierr);
      ierr = VecScatterCreate(z, is_z, gz, is_gz, &sum_ctxt);
      CHKERRQ(ierr);
      ierr = VecScatterCreateToAll(gz, &get_ctxt, &z);
      CHKERRQ(ierr);
    }

#if 0
    PetscOptionsHasName(PETSC_NULL,"-dump",&valid);
    if (valid) {
      // dump data structures for faster loading
      
      sprintf(tmp,"%s.p_ref.gz",filename);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,tmp,PETSC_BINARY_CREATE,&v);
      CHKERRQ(ierr);
      VecView(p_ref,v);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(v);CHKERRQ(ierr);
      
      sprintf(tmp,"%s.data.gz",filename);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,tmp,PETSC_BINARY_CREATE,&v);
      CHKERRQ(ierr);
      MatView(data,v);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(v);CHKERRQ(ierr);
    }
#endif
  }

  reduced = false;

  return 0;
}

// replace data matrix with reduced data matrix

#undef __FUNCT__
#define __FUNCT__ "Dataset::replaceEvents"
int Dataset::replaceEvents(Mat U, Mat V)
{
  Mat tmp;
  int ierr;

  // replace matrix

  tmp = data;
  data = U;
  ierr = MatDestroy(tmp);
  CHKERRQ(ierr);

  // adjust sizes

  ierr = MatGetSize(U, &nClasses, &nFeats);
  CHKERRQ(ierr);
  nNZeros = nClasses * nFeats;

  // replace e_ref

  ierr = VecDestroy(e_ref);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, nFeats, &e_ref);
  CHKERRQ(ierr);

  right = V;
  reduced = true;

  return 0;
}

// initialize reference probabilities

#undef __FUNCT__
#define __FUNCT__ "initializeDataset"
int initializeDataset(Dataset &d, PetscTruth transpose)
{
  int ierr, n, ia, *ii;
  PetscScalar zz, *xx;
  PetscTruth valid;
  char filename[LEN];
  const double minusone = -1.0;

  // simple smoothing

  if (!d.initialized)
  {
    ierr = VecDuplicate(d.z, &d.pmarg);
    CHKERRQ(ierr);
    double eps;
    PetscOptionsGetReal(PETSC_NULL, "-add", &eps, &valid);
    if (valid)
      VecShift(d.p_ref, eps);
  }

  // resample

  if (d.bootstrap)
  {

    int nprocs;
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    if (nprocs > 1)
      SETERRQ(PETSC_ERR_SUP, "MPI bootstrapping not supported");

    if (d.nContexts == 1)
    {

      // bootstrap events

      if (!d.initialized)
      {
        VecSum(d.p_ref, &zz);
        d.total = (int)zz;
        ierr = PetscMalloc(sizeof(int) * d.total, &d.counts);
        ii = d.counts;

        int p = 0;
        ierr = VecGetArray(d.p_ref, &xx);
        CHKERRQ(ierr);
        for (int i = 0; i < d.nClasses; i++)
          for (int j = 0; j < xx[i]; j++)
            ii[p++] = i;
        ierr = VecRestoreArray(d.p_ref, &xx);
        CHKERRQ(ierr);

        srand48(time(NULL));
      }

      // draw a random sample with replacement

      ierr = VecSet(d.p_ref, 0.0);
      CHKERRQ(ierr);

      ierr = VecGetArray(d.p_ref, &xx);
      CHKERRQ(ierr);
      ii = d.counts;

      for (int i = 0; i < d.total; i++)
      {
        double r = drand48() * d.total;
        int j = ii[(int)r];
        xx[j]++;
      }

      PetscLogFlops(d.nClasses);
      ierr = VecRestoreArray(d.p_ref, &xx);
      CHKERRQ(ierr);
    }
    else
    {

      SETERRQ(PETSC_ERR_SUP, "Bootstrapping contexts not supported");

      // bootstrap contexts

      if (!d.initialized)
      {

        sum_z(d.p_ref, &d);
        VecCopy(d.z, d.pmarg);
        ierr = VecDuplicate(d.pmarg, &d.stuff);
        CHKERRQ(ierr);
        ierr = VecCopy(d.pmarg, d.stuff);
        CHKERRQ(ierr);

        ierr = VecDuplicate(d.p_ref, &d.things);
        CHKERRQ(ierr);
        ierr = VecCopy(d.p_ref, d.things);
        CHKERRQ(ierr);

        srand48(time(NULL));
      }

      // draw a random sample with replacement

      Vec temp;
      ierr = VecDuplicate(d.pmarg, &temp);
      CHKERRQ(ierr);

      ierr = VecSet(temp, 0.0);
      CHKERRQ(ierr);

      ierr = VecGetArray(temp, &xx);
      CHKERRQ(ierr);

      ii = (int *)d.counts;

      for (int i = 0; i < d.nContexts; i++)
      {
        double r = drand48() * d.nContexts;
        xx[(int)r] += 1.0;
      }

      PetscLogFlops(d.nContexts);
      ierr = VecRestoreArray(temp, &xx);
      CHKERRQ(ierr);

      VecPointwiseMult(d.pmarg, d.stuff, temp);
      VecCopy(d.things, d.p_ref);
    }
  }

  // compute context frequencies

  PetscOptionsHasName(PETSC_NULL, "-uniform", &valid);
  if (valid)
  {
    // uniform marginal
    VecSet(d.pmarg, 1.0);
  }
  else
  {
    // pseudo-likelihood marginal
    sum_z(d.p_ref, &d);
    // get rid of this if bootstrap contexts!
    VecCopy(d.z, d.pmarg);
  }

  // compute probabilities of contexts

  VecSum(d.pmarg, &zz);
  zz = 1.0 / zz;
  VecScale(d.pmarg, zz);
  VecReciprocal(d.pmarg);

  // re-normalize event probabilities

  normalize(d.p_ref, &d);

  PetscTruth memd;
  PetscOptionsName("-memd", "MEMD estimation", "Estimate", &memd);
  if (memd)
    normalize(d.p0, &d);

  // entropy

  d.h0 = entropy(d.p_ref);

  // compute feature expectations

  MatMultTranspose(d.data, d.p_ref, d.e_ref);
  VecScale(d.e_ref, minusone);

// compute feature variances

#if SCALED

  double *sumsq;
  ierr = VecCreateSeq(

  ierr = PetscMalloc(sizeof(double)*d.nFeats,&sumsq);CHKERRQ(ierr);
  ierr = PetscMemzero(sumsq,sizeof(double)*d.nFeats);CHKERRQ(ierr);
  for(int i=0;i<d.nClasses;i++) {
    int ncols, *cols;
    double *vals;
    ierr = MatGetRow(d.data, i, &ncols, &cols, &vals);
    CHKERRQ(ierr);
    for (int j = 0; j < ncols; j++)
      sumsq[j] += vals[j]*vals[j};
    ierr = MatRestoreRow(d.data,i,&ncols,&cols,&vals);CHKERRQ(ierr);





  Vec tmp;





  ierr = VecDuplicate(d.e_ref,tmp); CHKERRQ(ierr);
  ierr = VecDot(d.e_ref,d.e_ref,tmp); CHKERRQ(ierr);
  VecScale(tmp,zz);

#endif

  if (!d.initialized) {

    double tmp = 0.0;
    d.penalty = NONE;

    // set up for Gaussian prior penalty

    PetscOptionsGetString(PETSC_NULL, "-variances", filename, LEN, &valid);
    if (valid)
    {

      int id;

      MPI_Comm_rank(PETSC_COMM_WORLD, &id);

      VecDuplicate(d.e_ref, &d.sigma);

      // read variances

      if (id == 0)
      {
        std::ifstream in(filename);
        if (!in)
          SETERRQ(PETSC_ERR_FILE_OPEN, "Error opening variances file");
        for (int i = 0; i < d.nFeats; ++i)
        {
          double var;
          in >> var;
          VecSetValue(d.sigma, i, var, INSERT_VALUES);
        }
        in.close();
      }

      VecAssemblyBegin(d.sigma);
      VecAssemblyEnd(d.sigma);

      d.penalty = L2;
    }

    // set up for Gaussian prior penalty (with one variance)

    PetscOptionsReal("-l2", "Variance for L2 penalty", "est", 0.0, &tmp, &valid);
    if (valid && tmp != 0.0)
    {
      VecDuplicate(d.e_ref, &d.sigma);
      tmp = 1.0 / tmp;
      VecSet(d.sigma, tmp);
      d.penalty = L2;
    }

    // set up for double exponential prior penalty (with one variance)

    PetscOptionsReal("-l1", "Variance for L1 penalty", "est", 0.0, &tmp, &valid);
    if (valid && tmp != 0.0)
    {
      int nprocs;
      MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
      if (nprocs > 1)
        SETERRQ(PETSC_ERR_SUP, "MPI L1 smoothing not supported yet");
      ierr = VecCreateSeq(PETSC_COMM_WORLD, d.nFeats * 2, &d.sigma);
      CHKERRQ(ierr);
      tmp = 1.0 / tmp;
      VecSet(d.sigma, tmp);
      // d.lambda = tmp;
      d.penalty = L1;
    }

    PetscOptionsGetString(PETSC_NULL, "-l1vars", filename, LEN, &valid);
    if (valid)
    {
      int nprocs;
      MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
      if (nprocs > 1)
        SETERRQ(PETSC_ERR_SUP, "MPI L1 smoothing not supported yet");
      ierr = VecCreateSeq(PETSC_COMM_WORLD, d.nFeats * 2, &d.sigma);
      CHKERRQ(ierr);

      // read variances

      std::ifstream in(filename);
      if (!in)
        SETERRQ(PETSC_ERR_FILE_OPEN, "Error opening variances file");
      for (int i = 0; i < d.nFeats * 2; ++i)
      {
        double var;
        in >> var;
        VecSetValue(d.sigma, i, var, INSERT_VALUES);
      }
      in.close();

      VecAssemblyBegin(d.sigma);
      VecAssemblyEnd(d.sigma);

      d.penalty = L1;
    }
  }

  // count active features
  
  VecGetLocalSize(d.e_ref,&n);
  VecGetArray(d.e_ref,&xx);
  
  ia = 0;
  for (int i = 0; i < n; i++) 
    if (xx[i] != 0.0)  
      ia++;
  
  VecRestoreArray(d.e_ref,&xx);
  
  MPI_Allreduce(&ia, &d.nActive, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  
 
  if (!d.initialized) {

    // transpose data matrix

    if (transpose)
    {
      Mat datat;
      ierr = MatTranspose(d.data, &datat);
      CHKERRQ(ierr);
      ierr = MatDestroy(d.data);
      CHKERRQ(ierr);
      d.data = datat;
    }
  }

  d.initialized = true;

  return 0;
}
