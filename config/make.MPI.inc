HYPREFLAGS = -g -Wall -DHAVE_CONFIG_H -DHYPRE_TIMING
HYPREINC   = -I${HYPRE_DIR}/include
LIBHYPRE   = -L${HYPRE_DIR}/lib -lHYPRE -llapack -lblas -lgfortran -lmpifort -lmpi -lm -lstdc++

PETSCFLAGS = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  
PETSCINC   = -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBPETSC   = -L${PETSC_DIR}/${PETSC_ARCH}/lib -Wl,-rpath,/usr/local -L/usr/local -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/7 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lpetsc -llapack -lblas -lpthread -lm -lstdc++ -ldl -lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

SLEPCFLAGS = -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3
SLEPCINC   = -I${SLEPC_DIR}/include -I${SLEPC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include
LIBSLEPC   = -Wl,-rpath,${SLEPC_DIR}/${PETSC_ARCH}/lib -L${SLEPC_DIR}/${PETSC_ARCH}/lib -L${PETSC_DIR}/${PETSC_ARCH}/lib -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/7 -L/usr/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu -Wl,-rpath,/usr/local/lib -lslepc -lpetsc -llapack -lblas -lpthread -lm -lstdc++ -ldl -lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl

PHGFLAGS   = -Wall -Wmissing-field-initializers -fopenmp -D__PHG__ 
PHGINC     = -I${PHG_DIR}/include
LIBPHG     = -L${PHG_DIR}/lib  -fopenmp -Wl,-rpath,${PHG_DIR}/lib -lphg  -lumfpack -lamd  -llapack -lblas  -lm  -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/7 -L/usr/lib/gcc/x86_64-linux-gnu/7/../../../x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/7/../../../../lib -L/lib/x86_64-linux-gnu -L/lib/../lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/7/../../.. -lgfortran -lm -lmpifort -lmpi -lquadmath  -lgfortran -lm -lmpifort -lmpi -lquadmath -Wl,-rpath,/usr/local/lib

LIBBLASLAPACK = 

CPP      = mpic++ -D__DEBUG__
CC       = mpicc  -D__DEBUG__
