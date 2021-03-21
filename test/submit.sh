#!/bin/sh
module purge
#module load compilers/intel-2018u4
#module load mpi/impi-2018u4
#module load apps/mkl-2018u4
module load mpi/oneapimpi-oneapi-2021.1.1
module load apps/mkl-oneapi-2021
filename=$1
#matrix=/share/home/hhxie/liyu/MatrixCollection/Andrews.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/c-65.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/Ga10As10H30.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/Ga3As3H12.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/Ga41As41H72.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/Si5H12.petsc.bin
#matrix=/share/home/hhxie/liyu/MatrixCollection/SiO2.petsc.bin
exename="./TestOPS.exe"
#exename="./TestOPS.exe -eps_monitor_conv -eps_nev 200 -eps_lobpcg_blocksize 40 -eps_lobpcg_restart 0.1"
#exename="./TestOPS.exe -gcge_compW_cg_max_iter 35 -file ${matrix}"
#exename="./TestOPS.exe -gcge_compW_cg_max_iter 60"
#exename="./TestOPS.exe -eps_monitor_conv -eps_nev 200 -eps_lobpcg_blocksize 40 -eps_lobpcg_restart 0.1 -file ${matrix}"
mpirun=/soft/apps/intel/oneapi_hpc_2021/mpi/2021.1.1/bin/mpiexec.hydra
#for nprocs in 1152
#for nprocs in 72
#for nprocs in 1152 36 72 144 288 576 
#for nprocs in 36
for nprocs in 576
#for nprocs in 1152
do 
#for nevConv in 100 200 400 800 1600
#for nevConv in 5000
for nevConv in 100
do
blockSize=20
#for blockSize in 50 40 30 20
#do
#nevMax=$[nevConv+blockSize+20]
nevMax=$[nevConv+nevConv]
#for nevMax in 120
#do 
	bsub -J $filename -q big \
     		-hostfile ./hosts/hosts${nprocs} \
     		-e OOerr -o ./results/$filename.${nevConv}.${nevMax}.${blockSize}.${nprocs}.$(date +%m%d.%H%M%S) \
     		"$mpirun $exename -nevConv ${nevConv} -nevMax ${nevMax} -blockSize ${blockSize}"
#done
#done 
done
done
