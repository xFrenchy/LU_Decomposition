# LU_Decomposition in OpenMP, MPI, & CUDA

###### Anthony Dupont, Trystan Kaes, Tobby Lie, Marcus Gallegos

## To run the Sequential version on heracles:

> g++ -O LU-Decomp_Seq.cpp -o LU-Decomp_Seq

> sbatch lu_decomp_seq_slurm.sh (size of matrix) (print yes = 1, print no = 0)

## To run the MPI version on heracles:

> mpicxx LU-Decomp-MPI.cpp -o mpi

> mpirun -print-rank-map -n (number of processes) -ppn (tasks per node) ./mpi (size of matrix) (print yes = 1, print no = 0)

> sbatch lu_mpi_slurm.sh (size of matrix) (print yes = 1, print no = 0)

The matrix size must be a multiple of the number of processes such as mpirun -print-rank-map -n 4 -ppn 2 ./mpi 8 1

If using the slurm script, edit --ntask and --ntask-per-node inside the file

## To run the OpenMP version on heracles:

> g++ -O -fopenmp LU-Decomp-OpenMP.cpp -o LU-Decomp-OpenMP

> sbatch lu_decomp_omp_slurm.sh (size of matrix) (number of threads to run program with) (print yes = 1, print no = 0)

## To run the CUDA version on heracles:

> ssh node18 nvcc -arch=sm_30 $PWD/LU-Decomp-CUDA.cu -o $PWD/LU-Decomp-CUDA

> sbatch lu_decomp_cuda_slurm.sh (size) (print yes = 1, print no = 0)
