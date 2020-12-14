/**
 *    @file  ops_multi_grid.c
 *   @brief  正交化操作 
 *
 *  正交化操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include    <math.h>

#include    "ops.h"


void DefaultVecFromItoJ(void **P_array, int level_i, int level_j, 
		void *vec_i, void *vec_j, void **vec_ws, struct OPS_ *ops)
{
    void *from_vec, *to_vec; int k = 0;
    if(level_i > level_j) {
    	/* 从粗层到细层，延拓，用P矩阵直接乘 */
		/* for level_i = 2 > level_j = 0 
		 * vec_ws = P1 vec_i 
		 * vec_j  = P0 vec_ws */       
       	for(k = level_i; k > level_j; --k) {
	  		if(k == level_i) {
	    	 	from_vec = vec_i;
	  		} else {
	    	 	from_vec = vec_ws[k];
	  		}
	  		if(k == level_j+1) {
	    	 	to_vec = vec_j;
	  		} else {
	    	 	to_vec = vec_ws[k-1];
	 		}
	  		ops->MatDotVec(P_array[k-1], from_vec, to_vec, ops);
      	}
    }
   	else if(level_i < level_j) {
   		/* 从细层到粗层，限制，用P^T矩阵乘 */
       	/* for level_i = 0 < level_j = 3 
		 * vec_ws = P0' vec_i 
		 * vec_ws = P1' vec_ws 
		 * vec_j  = P2' vec_ws */       
       	for(k = level_i; k < level_j; ++k) {
	  		if(k == level_i) {
	     		from_vec = vec_i;
	  		} else {
	     		from_vec = vec_ws[k];
	 	 	}
	 		if(k == level_j-1) {
	     		to_vec = vec_j;
	  		} else {
	    	 	to_vec = vec_ws[k+1];
	  		}
	  		ops->MatTransDotVec(P_array[k], from_vec, to_vec, ops);
       	}
    }
    else {
       	/* level_i == level_j */
    	ops->VecAxpby(1.0, vec_i, 0.0, vec_j, ops);
    }
    return;
}
void DefaultMultiVecFromItoJ(void **P_array, int level_i, int level_j, 
		void **multi_vec_i, void **multi_vec_j, int *startIJ, int *endIJ, 
		void ***multi_vec_ws, struct OPS_ *ops)
{
    void **from_vecs, **to_vecs;
    int k = 0, start[2], end[2];
    if(level_i > level_j) {
       	for(k = level_i; k > level_j; --k) {
	  		if(k == level_i) {
	    		from_vecs = multi_vec_i;
	    		start[0] = startIJ[0]; end[0] = endIJ[0];
	 		} else {
	   			from_vecs = multi_vec_ws[k];
	     		start[0] = 0; end[0] = endIJ[0]-startIJ[0];
	  		}
	  		if(k == level_j+1) {
	     		to_vecs = multi_vec_j;
	     		start[1] = startIJ[1]; end[1] = endIJ[1];
	  		} else {
	     		to_vecs = multi_vec_ws[k-1];
	     		start[1] = 0; end[1] = endIJ[0]-startIJ[0];
	  		}
	  		ops->MatDotMultiVec(P_array[k-1],from_vecs,to_vecs,start,end,ops);
       	}
    }
    else if (level_i < level_j) {
       	for(k=level_i; k<level_j; ++k) {
	  		if(k == level_i) {
	     		from_vecs = multi_vec_i;
	     		start[0] = startIJ[0]; end[0] = endIJ[0];
	  		} else {
	     		from_vecs = multi_vec_ws[k];
	     		start[0] = 0; end[0] = endIJ[0]-startIJ[0];
	  		}
	  		if(k == level_j-1) {
	     		to_vecs = multi_vec_j;
	     		start[1] = startIJ[1]; end[1] = endIJ[1];
	  		} else {
	     		to_vecs = multi_vec_ws[k+1];
	     		start[1] = 0; end[1] = endIJ[0]-startIJ[0];
	  		}
	  		ops->MatTransDotMultiVec(P_array[k],from_vecs,to_vecs,start,end,ops);
       	}
    }
    else {
       ops->MultiVecAxpby(1.0,multi_vec_i,0.0,multi_vec_j,startIJ,endIJ,ops);
    }
    return;
}
