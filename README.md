This is an attempt to make the
[Toolkit for Advanced Discriminative Modeling (TADM)](http://tadm.sourceforge.net/)
compile with current versions of [PETSc/Tao](https://www.mcs.anl.gov/petsc/)

**This is not working yet**

## To do

* tadm.cc
    * PetscGetTime()
* mle.cc
    * too many arguments to function TaoSetTolerances()
    * too few arguments to function TaoSetMonitor()
    * maxent_conv() → check for convergence
    * maxent_monitor()
    * check everything
* dataset.cc
    * check handling of options
* check for TODO in all source files
* after testing, modify code and remove defines in fix.h
