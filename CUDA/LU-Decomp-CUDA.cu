//-----------------------------------------------------------------------
// Parallel LU Decomposition - C++ CUDA
//-----------------------------------------------------------------------
//  Some features:
//  + None yet
//  Programming by Anthony Dupont, Trystan Kaes, Tobby Lie, Marcus Gallegos 
//  Update in 11/1/2020
// ----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

using namespace std;
#define TILE 16

//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& isPrint)
{
	bool isOK = true;
	
	if(argc < 2) 
	{
		cout << "Arguments:<X> [<Y>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y = 1: print the input/output matrix if X < 10" << endl;
		cout << "Y <> 1 or missing: does not print the input/output matrix" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}
		//is print the input/output matrix
		if (argc >=3)
			isPrint = (atoi(argv[2])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}


//------------------------------------------------------------------
//delete matrix matrix a[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **a,int n)
{
	delete[] a[0];
	delete[] a; 
}


//------------------------------------------------------------------------------------------------
//Fills matrix A with random values, upper and lower is filled with 0's except for their diagonals
//------------------------------------------------------------------------------------------------
void InitializeMatrices(float **&a, float **&lower, float **&upper, int size){
	a = new float*[size];
	a[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		a[i] = a[i-1] + size;
	lower = new float*[size];
	lower[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		lower[i] = lower[i-1] + size;
	upper = new float*[size];
	upper[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		upper[i] = upper[i-1] + size;
	
	for(int i = 0; i < size; ++i){
		for(int j = 0; j < size; ++j){
			a[i][j] = (rand() % 11) + 1;
			if(i == j){
				//we fill the diagonal with 1's
				lower[i][j] = 1;
				upper[i][j] = 1;	
			}
			else{
				lower[i][j] = 0;
				upper[i][j] = 0;
			}
		}
	}
}


//------------------------------------------------------------------
//Print the matrix that was passed to it
//------------------------------------------------------------------
void PrintMatrix(float **matrix, int size) 
{
	for (int i = 0 ; i < size ; i++){
		for (int j = 0 ; j < size ; j++){
			printf("%.2f\t", matrix[j][i]);
		}
		printf("\n");
	}
}


__global__ void LUDecomp(float **a, float **lower, float **upper, int size){
	//Placeholder junk
	int Row = blockIdx.x*TILE + threadIdx.x;
	int Col = blockIdx.y*TILE + threadIdx.y;
}


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
		cout<< "Lower:" << endl;
		PrintMatrix(lower,n); 

		cout<< "Upper:" << endl;
		PrintMatrix(upper,n); 
	}

	cout<< "LU Decomposition ran in " << setiosflags(ios::fixed) << setprecision(2) << (runtime)/float(CLOCKS_PER_SEC) << " seconds\n";

	// TODO: Then we print out the actual solution we got from the back substitution.
	
	cudaFree(d_lower);
	cudaFree(d_upper);
	cudaFree(d_a);

	DeleteMatrix(upper,n);	
	DeleteMatrix(lower,n);	
	DeleteMatrix(a,n);	

	return 0;
}
