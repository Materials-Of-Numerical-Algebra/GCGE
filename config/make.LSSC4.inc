HYPREFLAGS = -g -O2 -fopenmp
HYPREINC   = -I/soft/apps/phg/intel-2017u4/impi-2017u3/hypre/include
LIBHYPRE   = -L/soft/apps/phg/intel-2017u4/impi-2017u3/hypre/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/hypre/lib \
	     -lHYPRE 

PETSCFLAGS = -g -O2 -fopenmp
PETSCINC   = -I/soft/apps/phg/intel-2017u4/impi-2017u3/petsc/include
LIBPETSC   = -L/soft/apps/phg/intel-2017u4/impi-2017u3/petsc/lib \
	     -L/soft/apps/phg/intel-2017u4/impi-2017u3/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/petsc/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/lib \
	     -lpetsc \
	     -lsmumps -ldmumps -lmumps_common -lpord -lptesmumps -lptscotch -lptscotcherrexit -lesmumps -lscotch -lscotcherrexit \
	     -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lmkl_lapack95_lp64 \
	     -lparmetis -lmetis -lGKlib \
	     -lmpifort -lmpi -lmpigi -lrt -lpthread -lifport -lifcoremt_pic -limf -lsvml -lm -lipgo -liomp5 -lirc -lgcc_s -lirc_s -lquadmath -lstdc++ -ldl

SLEPCFLAGS = -g -O2 -fopenmp
SLEPCINC   = -I/share/home/hhxie/liyu/soft/slepc/include $(PETSCINC)
LIBSLEPC   = -L/share/home/hhxie/liyu/soft/slepc/lib \
	     -Wl,-rpath,/share/home/hhxie/liyu/soft/slepc/lib \
	     -lslepc \
	     $(LIBPETSC)

PHGFLAGS   = -g -O2 -fopenmp
PHGINC     = -I/soft/apps/phg/intel-2017u4/impi-2017u3/phg-0.9.6/include
LIBPHG     = -L/soft/apps/phg/intel-2017u4/impi-2017u3/phg-0.9.6/lib \
	     -L/soft/apps/phg/intel-2017u4/impi-2017u3/petsc/lib \
	     -L/soft/apps/phg/intel-2017u4/impi-2017u3/hypre/lib \
	     -L/soft/apps/phg/intel-2017u4/impi-2017u3/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/phg-0.9.6/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/petsc/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/hypre/lib \
	     -Wl,-rpath,/soft/apps/phg/intel-2017u4/impi-2017u3/lib \
	     -lphg -lpetsc -lHYPRE \
	     -lsuperlu_dist \
	     -lminres \
	     -lBLOPEX \
	     -lparpack \
	     -lmpfr -lmatheval

LIBBLASLAPACK = -lmkl_lapack95_lp64 -lmkl_blas95_lp64 -mkl=cluster

CPP      = mpicxx
CC       = mpicc


#-lsmumps -ldmumps -lmumps_common \
#-lptesmumps -lesmumps \
#-lptscotch -lptscotcherrexit -lptscotcherr -lptscotchparmetis \
#-lscotch -lscotcherrexit -lscotchmetis -lscotcherr \
#-lparmetis -lmetis \
# -lpapi -lGKlib \
#-liomp5
