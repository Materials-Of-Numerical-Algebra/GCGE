# Project: TestOPS
# Makefile created by Dev-C++ 5.11

#include ../config/make.PHG.inc
#include ../config/make.MPI.inc
#include ../config/make.LSSC4.inc
include ../config/make.LSSC4.oneapi2021.inc
#memwatch.o 
OBJ      = main.o mat_convert.o test_app_ccs.o test_app_lapack.o test_app_phg.o test_app_hypre.o test_app_slepc.o test_app_pas_lapack.o test_app_pas_ccs.o test_app_pas_slepc.o test_eig_sol_gcg.o test_eig_sol_pas.o test_lin_sol.o test_multi_grid.o test_multi_vec.o test_orth.o test_vec.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_grid.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_ccs.o ../app/app_lapack.o ../app/app_matlab.o ../app/app_pas.o ../app/app_slepc.o ../app/app_phg.o ../app/app_hypre.o ../src/ops_eig_sol_pas.o get_mat_phg.o 
LINKOBJ  = main.o mat_convert.o test_app_ccs.o test_app_lapack.o test_app_phg.o test_app_hypre.o test_app_slepc.o test_app_pas_lapack.o test_app_pas_ccs.o test_app_pas_slepc.o test_eig_sol_gcg.o test_eig_sol_pas.o test_lin_sol.o test_multi_grid.o test_multi_vec.o test_orth.o test_vec.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_grid.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_ccs.o ../app/app_lapack.o ../app/app_matlab.o ../app/app_pas.o ../app/app_slepc.o ../app/app_phg.o ../app/app_hypre.o ../src/ops_eig_sol_pas.o get_mat_phg.o 
LIBS     = $(LIBPHG) $(LIBSLEPC) $(LIBPETSC) $(LIBHYPRE) $(LIBBLASLAPACK)
INCS     = -I../src -I../app
CXXINCS  = 
BIN      = TestOPS.exe
CXXFLAGS = $(CXXINCS) -O2 -Wall -fPIC -g
CFLAGS   = $(INCS) -O2 -Wall -fPIC -g -qopenmp #-fopenmp -DMEMWATCH -DMW_STDIO ##
RM       = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ) $(BIN) memwatch.o

$(BIN): $(OBJ) memwatch.o
	$(CC) $(LINKOBJ) -o $(BIN) $(LIBS)

main.o: main.c
	$(CC) -c main.c -o main.o $(CFLAGS)

get_mat_phg.o: get_mat_phg.c
	$(CC) -c get_mat_phg.c -o get_mat_phg.o $(CFLAGS) $(PHGFLAGS) $(PHGINC)

mat_convert.o: mat_convert.c
	$(CC) -c mat_convert.c -o mat_convert.o $(CFLAGS) $(PHGFLAGS) $(PHGINC) $(PETSCFLAGS) $(PETSCINC) $(HYPREFLAGS) $(HYPREINC)

memwatch.o: memwatch.c
	$(CC) -c memwatch.c -o memwatch.o $(CFLAGS)

test_app_ccs.o: test_app_ccs.c
	$(CC) -c test_app_ccs.c -o test_app_ccs.o $(CFLAGS)

test_app_lapack.o: test_app_lapack.c
	$(CC) -c test_app_lapack.c -o test_app_lapack.o $(CFLAGS)

test_app_slepc.o: test_app_slepc.c
	$(CC) -c test_app_slepc.c -o test_app_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

test_app_phg.o: test_app_phg.c
	$(CC) -c test_app_phg.c -o test_app_phg.o $(CFLAGS) $(PHGFLAGS) $(PHGINC)

test_app_hypre.o: test_app_hypre.c
	$(CC) -c test_app_hypre.c -o test_app_hypre.o $(CFLAGS) $(HYPREFLAGS) $(HYPREINC)

test_app_pas_lapack.o: test_app_pas_lapack.c
	$(CC) -c test_app_pas_lapack.c -o test_app_pas_lapack.o $(CFLAGS)

test_app_pas_ccs.o: test_app_pas_ccs.c
	$(CC) -c test_app_pas_ccs.c -o test_app_pas_ccs.o $(CFLAGS)

test_app_pas_slepc.o: test_app_pas_slepc.c
	$(CC) -c test_app_pas_slepc.c -o test_app_pas_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

test_eig_sol_gcg.o: test_eig_sol_gcg.c
	$(CC) -c test_eig_sol_gcg.c -o test_eig_sol_gcg.o $(CFLAGS)

test_eig_sol_pas.o: test_eig_sol_pas.c
	$(CC) -c test_eig_sol_pas.c -o test_eig_sol_pas.o $(CFLAGS)

test_lin_sol.o: test_lin_sol.c
	$(CC) -c test_lin_sol.c -o test_lin_sol.o $(CFLAGS)

test_multi_grid.o: test_multi_grid.c
	$(CC) -c test_multi_grid.c -o test_multi_grid.o $(CFLAGS)

test_multi_vec.o: test_multi_vec.c
	$(CC) -c test_multi_vec.c -o test_multi_vec.o $(CFLAGS)

test_orth.o: test_orth.c
	$(CC) -c test_orth.c -o test_orth.o $(CFLAGS)

test_vec.o: test_vec.c
	$(CC) -c test_vec.c -o test_vec.o $(CFLAGS)

../src/ops.o: ../src/ops.c
	$(CC) -c ../src/ops.c -o ../src/ops.o $(CFLAGS)

../src/ops_eig_sol_gcg.o: ../src/ops_eig_sol_gcg.c
	$(CC) -c ../src/ops_eig_sol_gcg.c -o ../src/ops_eig_sol_gcg.o $(CFLAGS)

../src/ops_eig_sol_pas.o: ../src/ops_eig_sol_pas.c
	$(CC) -c ../src/ops_eig_sol_pas.c -o ../src/ops_eig_sol_pas.o $(CFLAGS)

../src/ops_lin_sol.o: ../src/ops_lin_sol.c
	$(CC) -c ../src/ops_lin_sol.c -o ../src/ops_lin_sol.o $(CFLAGS)

../src/ops_multi_grid.o: ../src/ops_multi_grid.c
	$(CC) -c ../src/ops_multi_grid.c -o ../src/ops_multi_grid.o $(CFLAGS)

../src/ops_multi_vec.o: ../src/ops_multi_vec.c
	$(CC) -c ../src/ops_multi_vec.c -o ../src/ops_multi_vec.o $(CFLAGS)

../src/ops_orth.o: ../src/ops_orth.c
	$(CC) -c ../src/ops_orth.c -o ../src/ops_orth.o $(CFLAGS)

../app/app_ccs.o: ../app/app_ccs.c
	$(CC) -c ../app/app_ccs.c -o ../app/app_ccs.o $(CFLAGS)

../app/app_lapack.o: ../app/app_lapack.c
	$(CC) -c ../app/app_lapack.c -o ../app/app_lapack.o $(CFLAGS)

../app/app_matlab.o: ../app/app_matlab.c
	$(CC) -c ../app/app_matlab.c -o ../app/app_matlab.o $(CFLAGS)

../app/app_pas.o: ../app/app_pas.c
	$(CC) -c ../app/app_pas.c -o ../app/app_pas.o $(CFLAGS)

../app/app_slepc.o: ../app/app_slepc.c
	$(CC) -c ../app/app_slepc.c -o ../app/app_slepc.o $(CFLAGS) $(SLEPCFLAGS) $(SLEPCINC)

../app/app_phg.o: ../app/app_phg.c
	$(CC) -c ../app/app_phg.c -o ../app/app_phg.o  $(CFLAGS) $(PHGFLAGS) $(PHGINC)

../app/app_hypre.o: ../app/app_hypre.c
	$(CC) -c ../app/app_hypre.c -o ../app/app_hypre.o  $(CFLAGS) $(HYPREFLAGS) $(HYPREINC)

