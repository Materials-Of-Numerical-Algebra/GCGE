#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
	/* ONLY for 2 processors */
	MPI_Init(&argc, &argv);
	int rank, i, k;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int nvec = 2, nglobal = 3, nlocal = rank+1;
	double  send_buff[6] = {1,10,100,2,20,200}; // 0-rank 1*2 1-rank 2*2
	double  recv_buff[6] = {0,0,0,0,0,0}; // 0-rank 1*2 1-rank 2*2
	if (rank == 0) {
		for (i = 0; i < nglobal; ++i) {
			for (k = 0; k < nvec; ++k) 
				printf("%.2f\t",send_buff[i+nglobal*k]);
			printf("\n");
		}
	}

	int cnts[2] = {1,2}, dsps[2] = {0,1};
	MPI_Datatype ROW_TYPE0;
	MPI_Request request[2];
	/* scatter send_buff to 0-rank and 1-rank in recv_buff */
	for (i = 0; i < 2; ++i) {
		if (rank == 0) {
			MPI_Type_vector(nvec, cnts[i], nglobal, MPI_DOUBLE, &ROW_TYPE0);
			MPI_Type_commit(&ROW_TYPE0);
			MPI_Isend(send_buff+dsps[i], 1, ROW_TYPE0, i, i, MPI_COMM_WORLD, request+i);
			MPI_Type_free(&ROW_TYPE0);
		}
		if (rank == i) {
			MPI_Irecv(recv_buff, nvec*cnts[i], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, request+i);
			MPI_Wait(request+i,MPI_STATUS_IGNORE);
		}
	}
	if (rank == 1) {
		printf("scatter send_buff to %d-rank in recv_buff\n",rank);
		for (i = 0; i < nlocal; ++i) {
			for (k = 0; k < nvec; ++k) 
				printf("%.2f\t",recv_buff[i+nlocal*k]);
			printf("\n");
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* gather send_buff to 0-rank in recv_buff */
	if (rank == 0) {
		send_buff[0] =  1; send_buff[1] =   2;
	} 
	else if (rank == 1) {
		send_buff[0] = 10; send_buff[2] =  20;
		send_buff[1] =100; send_buff[3] = 200;
	}
	recv_buff[0] = 0; recv_buff[3] = 0;
	recv_buff[1] = 0; recv_buff[4] = 0;
	recv_buff[2] = 0; recv_buff[5] = 0;
	for (i = 0; i < 2; ++i) {
		if (rank == i) {
			MPI_Isend(send_buff, nvec*cnts[i], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, request+i);
		}
		if (rank == 0) {
			MPI_Type_vector(nvec, cnts[i], nglobal, MPI_DOUBLE, &ROW_TYPE0);
			MPI_Type_commit(&ROW_TYPE0);
			MPI_Irecv(recv_buff+dsps[i], 1, ROW_TYPE0, i, i, MPI_COMM_WORLD, request+i);
			MPI_Wait(request+i,MPI_STATUS_IGNORE);
			MPI_Type_free(&ROW_TYPE0);
		}
	}
	if (rank == 0) {
		printf("gather send_buff to 0-rank in recv_buff\n");
		for (i = 0; i < nglobal; ++i) {
			for (k = 0; k < nvec; ++k) 
				printf("%.2f\t",recv_buff[i+nglobal*k]);
			printf("\n");
		}
	}

#if 0

	MPI_Datatype ROW_TYPE0, ROW_TYPE;
	MPI_Type_vector(nvec, 1, nglobal, MPI_DOUBLE, &ROW_TYPE0);
	MPI_Type_create_resized(ROW_TYPE0, 0, 1*sizeof(MPI_DOUBLE), &ROW_TYPE);
	MPI_Type_commit(&ROW_TYPE);
	MPI_Scatterv(send_buff, cnts, dsps, ROW_TYPE, 
			recv_buff, nvec*nlocal, MPI_DOUBLE,
			0, MPI_COMM_WORLD);

	if (rank == 1) {
		printf("[%d] nvec*nlocal = %d\n",rank,nvec*nlocal);
		MPI_Send(send_buff, nvec*(nlocal-1), MPI_DOUBLE, 0, 12345, MPI_COMM_WORLD);
	}
	else if (rank == 0) {
		MPI_Recv(recv_buff, 1, ROW_TYPE, 1, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Gatherv(send_buff, nlocal*nvec, MPI_DOUBLE, 
			recv_buff, cnts, dsps,
			ROW_TYPE, 0, MPI_COMM_WORLD);
#endif

	MPI_Finalize();
	return 0;
}
