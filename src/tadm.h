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

// $Id: tadm.h,v 1.2 2005/09/26 19:40:58 jasonbaldridge Exp $

// Copyright (c) 2001-2002 Robert Malouf

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "tao.h"

#undef SVD        // use SVD-based regularization

// Size for small strings
const int LEN = 256;

// Size of call to VecSetValues and MatSetValues
const int BATCH_SIZE = 10000;

// penalty types
const int NONE = 0;
const int L1 = 1;
const int L2 = 2;
const int BRIDGE = 3;

// Dataset stuff

class Dataset {

 public:
  
  int nClasses;   // number of classes in dataset
  int nFeats;     // number of features (actually, highest feature number + 1)
  int nActive;    // number of features with non-zero expected value
  int nNZeros;    // number of non-zero values
  Mat data;       // data matrix (rows=classes, cols=features)

  int nContexts;  // number of conditioning contexts 
  int *context;   // index of conditioning contexts

  bool correct;   // do we need a correction constant?
  double c;       // correction constant 

  Vec p_ref;      // reference probabilities P(y|x) 

  Vec p0;         // prior probabilities (for MEMD estimation)

  PetscScalar h0; // entropy of reference distribution 

  Vec e_ref;      // EVs of features given p_ref
  Vec v_ref;      // variances of features given p_ref

  Vec pmarg;      // marginal probabilities of contexts P(x)

  int penalty;    // compute penalized likelihood?
  Vec sigma;      // variances for penalty
  double lambda,q;

  int bootstrap;  // bootstrap replicates?
  int *counts;    // reference counts, used for computing P(y|x)
  int total;
  Vec stuff;
  Vec things;

  bool initialized; 

  Vec z;
  Vec gz;
  IS is_z;
  IS is_gz;
  VecScatter sum_ctxt,get_ctxt;
  int firstContext;
  int lastContext;

  bool reduced;
  Mat right;

  Dataset();

  int readEvents(char *filename);
  int replaceEvents(Mat U, Mat V);

};


int initializeDataset(Dataset &d, PetscTruth transpose);

// Model stuff

class Model {
 public:
  Vec q;
  Vec params;
  char *file_out;
  PetscTruth bin_file_out;
  Dataset *d;
  double h,kl;

  Model(Dataset *d);
};

int writeParams(Model *m, int i);
int variance (Dataset *d, Vec x, Vec p, Vec var);
int standardError (Model &m, Vec se);
int waldStatistic (Model &m, Vec wald);

// MLE stuff

class mle {
 public:
  Model *m;
  Dataset *d;
  Vec penalty;
  Vec params2,p1,p2,g1,g2;
  char *method;
  double fatol, frtol, lastf;
  PetscTruth monitor;
  int checkpoint, max_it;
  int fg,its;
  TAO_SOLVER tao;
  TAO_APPLICATION tao_appl;

  mle(Model *m);
};

int estimate_params(mle *e);
TaoTerminateReason maxent_conv (int i, double f, mle *e);

// Logging stuff

extern int NORMALIZE_EVENT, LIKELIHOOD_EVENT, IIS_EVENT, UPDATE_EVENT, 
  SUMZ_EVENT, ENTROPY_EVENT, LOAD_STAGE, INIT_STAGE, ESTIMATE_STAGE, 
  FINAL_STAGE,VECEXP_EVENT, MEASURE_EVENT, SCAN_EVENT, READ_EVENT;

// IIS stuff

int estimate_params_iis(mle *e);
int estimate_params_gis(mle *e);
int estimate_params_perceptron(mle *e);

// Probability stuff

PetscScalar entropy(Vec p);
int sum_z(Vec p, Dataset *d);
int VecExp(Vec x);
int normalize(Vec x, Dataset *d); 
int likelihood (Vec x, double *f, Vec g, mle *e);
int log_likelihood(Vec x, Vec y, double *f, double *h);
double aic(Vec x);
double bic(Vec x, int n);

// return larger of x and y

template<class type> inline
const type& max(const type& x, const type& y)
{       
  return (x < y ? y : x);
}

// return smaller of x and y

template<class type> inline
const type& min(const type& x, const type& y)
{       
  return (x > y ? y : x);
}

