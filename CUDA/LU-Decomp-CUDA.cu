/** @file
  * Name: Parallel LU Decomposition - CUDA Version
  * Authored by: Team Segfault
  * Description: This program performs Lower/Upper decomposition on a square matrix and 
  * subsequently solves the associated system of equations with Forward and Backward substitution.
  * Implementation Date: 11/23/2020
*/

#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.cuh"
#include "utilities.cuh"

//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]){
	srand(time(NULL));	//set the seed
	
	//Matrices
	float **a, **lower, **upper;

	float *d_a, *d_lower, *d_upper; //device pointers
	
	int	n,isPrintMatrix;
	double runtime;

	//Get program input
	if (GetUserInput(argc,argv,n,isPrintMatrix)==false) return 1;

	//Initialize the matrices
	InitializeMatrices(a, lower, upper, n);

	printf("A:\n");
	PrintMatrix(a,n); 
	// printf("\nLower:\n");
	// PrintMatrix(lower,n);
	// printf("\nUpper:\n");
	// PrintMatrix(upper,n);

	//Get start time
	runtime = clock()/(float)CLOCKS_PER_SEC;

	//Declare grid size and block size
	int numblock = n/TILE + ((n%TILE)?1:0);
	dim3 dimGrid(numblock,numblock);	
	dim3 dimBlock(TILE,TILE);	

	//Allocate memory on device
	cudaMalloc((void**)&d_a, n*n*sizeof(float));
	cudaMalloc((void**)&d_lower, n*n*sizeof(float));
	cudaMalloc((void**)&d_upper, n*n*sizeof(float));

	//Copy data to the device
	cudaMemcpy(d_a, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

	/*	Don't think we need to copy this stuff yet. 
		We will pull the upper and lower matrices after we are computing.
		Wait... I think we actually need the diagnol 1's. nevermind.
		This will be uncommented later. */
	// cudaMemcpy(d_lower, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_upper, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	

	//Compute the LU Decomposition
	LUDecomp<<<dimGrid,dimBlock>>>(d_a, d_lower, d_upper, n);

	cudaThreadSynchronize();

	/Get results from the device
	cudaMemcpy(lower[0],d_lower, n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(upper[0],d_upper, n*n*sizeof(float),cudaMemcpyDeviceToHost);
	
	runtime = clock() - runtime; //Make note of LU Decomp


	// TODO: Write the substitution function. That is a future problem.
	
	//Print the output matrix
	if (isPrint==1)
	{
		printf("Lower:\n");
		PrintMatrix(lower,n); 

		printf("Upper:\n");
		PrintMatrix(upper,n); 
	}

	printf("LU Decomposition ran in %.2f seconds\n", (runtime)/float(CLOCKS_PER_SEC));
	
	cudaFree(d_lower);
	cudaFree(d_upper);
	cudaFree(d_a);

	DeleteMatrix(upper,n);	
	DeleteMatrix(lower,n);	
	DeleteMatrix(a,n);	

	return 0;
}
