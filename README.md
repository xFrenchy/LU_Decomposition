# LU_Decomposition in OpenMP, MPI, & CUDA
###### Anthony Dupont, Trystan Kaes, Tobby Lie, Marcus Gallegos

## To run the Sequential version on heracles:

> g++ -O LU-Decomp_Seq.cpp -o LU-Decomp_Seq

> sbatch lu_decomp_seq_slurm.sh (size of matrix) (print yes = 1, print no = 0)

## To run the MPI version on heracles:

> mpirun -print-rank-map -n # -ppn # ./ProgramName # 1

Replace # with a number to represent the amount of processes, tasks per node, and matrix size respectively

## To run the OpenMP version of heracles:

> g++ -O -fopenmp LU-Decomp-OpenMP.cpp -o LU-Decomp-OpenMP

> sbatch lu_decomp_omp_slurm.sh (size of matrix) (number of threads to run program with) (print yes = 1, print no = 0)
