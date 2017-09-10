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
// *** model
// ***

// $Id: model.cc,v 1.3 2007/09/03 19:16:48 jasonbaldridge Exp $

//  Copyright (c) 2001-2002 Robert Malouf

#include "tadm.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

// model constructor

#undef __FUNCT__
#define __FUNCT__ "model::model"
Model::Model(Dataset *ds)
{
  char *filename;
  PetscTruth valid;

  d = ds;

  PetscMalloc(LEN*sizeof(char),&filename);

  PetscOptionsGetString(PETSC_NULL,"-params_out",filename,LEN,&valid);
  if (valid == PETSC_TRUE) {
    file_out = filename;
    bin_file_out = PETSC_FALSE;
    PetscPrintf(PETSC_COMM_WORLD,"Params out = %s\n", filename);
  } else {
    PetscOptionsGetString(PETSC_NULL,"-bin_params_out",filename,LEN,&valid);
    if (valid == PETSC_TRUE) {
      file_out = filename;
      bin_file_out = PETSC_TRUE;
      PetscPrintf(PETSC_COMM_WORLD,"Binary params out = %s\n", filename);
    } else {
      PetscFree(filename);
      file_out = NULL;
    }
  }

  // set up parameters

  VecDuplicate(d->e_ref,&params);
  VecSet(params,0.0);

  // set up probabilities

  VecDuplicate(d->p_ref,&q);
  VecSet(q,0.0);
}

// write out parameter vector

#undef __FUNCT__
#define __FUNCT__ "writeParams"
int 
writeParams(Model *m, int i)
{
  int ierr;
  PetscViewer v;
  char name[LEN];

  if (!m->file_out) return 0;

  // make filename

  if (i >= 0) {
    sprintf(name,"%s.%d",m->file_out,i);
  } else {
    sprintf(name,"%s",m->file_out);
  }

  if (m->bin_file_out) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_WRITE,
				 &v);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, name, &v);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(v,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  }

  // write vector

  if (m->d->reduced) {
    
    // translate parameter vector back into original space

    int ierr,n,d;
    Vec x;
    
    ierr = MatGetSize(m->d->right,&n,&d);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
    ierr = MatMult(m->d->right,m->params,x);CHKERRQ(ierr);
    ierr = VecView(x, v);CHKERRQ(ierr);
    ierr = VecDestroy(x);

  } else {

    ierr = VecView(m->params, v);CHKERRQ(ierr);

  }

  // close

  PetscViewerDestroy(v);

  // write correction feature

  if (m->d->correct) {
    std::ofstream out("gis.corr");
    out << m->d->c << std::endl;
    out.close();
  }

  return 0;
}

