/**
 *    @file  app_lapack.h
 *   @brief  app of lapack 
 *
 *  lapack的操作接口, 针对稠密矩阵
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */
#ifndef  _APP_LAPACK_H_
#define  _APP_LAPACK_H_

#include	"ops.h"

typedef struct LAPACKMAT_ {
	double *data; int nrows; int ncols; int ldd;
} LAPACKMAT;
typedef LAPACKMAT LAPACKVEC;

void OPS_LAPACK_Set  (struct OPS_ *ops);

/* BLAS */
#define dasum FORTRAN_WRAPPER(dasum)
double dasum(int *n, double *dx, int *incx);
#define daxpy FORTRAN_WRAPPER(daxpy)
int daxpy(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
#define dcopy FORTRAN_WRAPPER(dcopy)
int dcopy(int *n, double *dx, int *incx, double *dy, int *incy);
#define ddot FORTRAN_WRAPPER(ddot)
double ddot(int *n, double *dx, int *incx, double *dy, int *incy);
#define dgemm FORTRAN_WRAPPER(dgemm)
int dgemm(char *transa, char *transb, int *m, int *n, int *k,
		double *alpha, double *a, int *lda, 
		               double *b, int *ldb,
		double *beta , double *c, int *ldc);
#define dgemv FORTRAN_WRAPPER(dgemv)
int dgemv(char *trans, int *m, int *n,
		double *alpha, double *a, int *lda,
		               double *x, int *incx,
		double *beta , double *y, int *incy);
#define dlamch FORTRAN_WRAPPER(dlamch)
double dlamch(char *cmach);			
#define idamax FORTRAN_WRAPPER(idamax)
int idamax(int  *n, double *dx, int *incx);
#define dscal FORTRAN_WRAPPER(dscal)
int dscal(int *n, double *da, double *dx, int *incx);
#define dsymm FORTRAN_WRAPPER(dsymm)
int dsymm(char *side, char *uplo, int *m, int *n,
		double *alpha, double *a, int *lda, 
	 	               double *b, int *ldb, 
		double *beta , double *c, int *ldc);
#define dsymv FORTRAN_WRAPPER(dsymv)
int dsymv(char *uplo, int *n, 
		double *alpha, double *a, int *lda, 
		               double *x, int *incx, 
		double *beta , double *y, int *incy);
/* LAPACK */
/* DGEQP3 computes a QR factorization with column pivoting of 
 * a matrix A:  A*P = Q*R  using Level 3 BLAS 
 * LWORK >= 2*N+( N+1 )*NB, where NB is the optimal blocksize */
#define dgeqp3 FORTRAN_WRAPPER(dgeqp3)
int dgeqp3(int *m, int *n, double *a, int *lda, int *jpvt,
	double *tau, double *work, int *lwork, int *info);
/* DORGQR generates an M-by-N real matrix Q with 
 * orthonormal columns 
 * K is the number of elementary reflectors whose product 
 * defines the matrix Q. N >= K >= 0.
 * LWORK >= N*NB, where NB is the optimal blocksize */
#define dorgqr FORTRAN_WRAPPER(dorgqr)
int dorgqr(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);
/* The length of the array WORK.  LWORK >= 1, when N <= 1;
 * otherwise 8*N.
 * For optimal efficiency, LWORK >= (NB+3)*N,
 * where NB is the max of the blocksize for DSYTRD and DORMTR
 * returned by ILAENV. */
/* RQ factorization */
#define dgerqf FORTRAN_WRAPPER(dgerqf)
int dgerqf(int *m, int *n, double *a, int *lda, 
	double *tau, double *work, int *lwork, int *info);
#define dorgrq FORTRAN_WRAPPER(dorgrq)
int dorgrq(int *m, int *n, int *k, double *a, int *lda, 
	double *tau, double *work, int *lwork, int *info);
#define dsyev FORTRAN_WRAPPER(dsyev)
int dsyev(char *jobz, char *uplo, int *n, 
	double *a, int *lda, double *w, 
	double *work, int *lwork, int *info);
#define dsyevx FORTRAN_WRAPPER(dsyevx)
int dsyevx(char *jobz, char *range, char *uplo, int *n, 
	double *a, int *lda, double *vl, double *vu, int *il, int *iu, 
	double *abstol, int *m, double *w, double *z, int *ldz, 
	double *work, int *lwork, int *iwork, int *ifail, int *info);
	
#endif  /* -- #ifndef _APP_LAPACK_H_ -- */
