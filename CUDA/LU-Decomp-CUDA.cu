/** @file
  * Name: Parallel LU Decomposition - CUDA Version
  * Authored by: Team Segfault
  * Description: This program performs Lower/Upper decomposition on a square matrix and 
  * subsequently solves the associated system of equations with Forward and Backward substitution.
  * Implementation Date: 11/23/2020
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

void DeleteMatrix(float**,int);
void PrintMatrix(float **, int);
void InitializeMatrices(float **&, float **&, float **&, int);
bool GetUserInput(int, char *[], int&,int&);
void sequentialLUdecomposition(float**, float** &, int);

void cudaLUDecomp(float **a, float *d_a, float *d_lower, float *d_upper, int thicness){

	float pivot;
	int i, numBlocks, numThreads;

	for(i = 0; i < thicness; ++pivot_i){

		pivot = -1.0/a[i][i];	

		numBlocks = thicness-i;
		numThreads = thicness-i; // Since all of these are square these are the same.

		dim3 dimGrid(numBlocks,1);	
		dim3 dimBlock(numThreads,1);	

		RowOperation<<<dimGrid,dimBlock>>>(d_a, d_lower, d_upper, pivot, i, thicness);
	}

	//Get results from the device
	cudaMemcpy(lower[0],d_lower, n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(upper[0],d_upper, n*n*sizeof(float),cudaMemcpyDeviceToHost);
}


__global__ void RowOperation(float *a, float *lower, float *upper, int pivot, int i, int thicness){
	if(blockIdx.x * blockDim.x  + threadIdx.x == 0) {// Lets get this out of the way
		lower[ k*thicness + k ] = 1; //lower[i][i] = 1
	}
	
    // Lets get some readability up in here.
    int k = blockIdx.x + i + 1; // local_k
	int j = threadIdx.x + i; // local_j
	
	if( k < thicness && j < thicness){

		float temp = pivot*a[k*thicness + i];

		lower[k*thicness + i] = a[k*thicness + i]/a[i*thicness + i];

		upper[k*thicness + j] = a[k*thicness + j] + temp * a[i*thicness + j];


	}
    
}


//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]){
	srand(time(NULL));	//set the seed
	
	//Matrices
	float **a, **lower, **upper;
	//Device pointers
	float *d_a, *d_lower, *d_upper;
	
	int	n,isPrintMatrix;
	float runtime;

	//Get program input
	if (!GetUserInput(argc,argv,n,isPrintMatrix)) return 1;

	//Initialize the matrices
	InitializeMatrices(a, lower, upper, n);

	//Get start time
	runtime = clock()/(float)CLOCKS_PER_SEC;

	cudaMalloc((void**)&d_a, n*n*sizeof(float));
	cudaMalloc((void**)&d_lower, n*n*sizeof(float));
	cudaMalloc((void**)&d_upper, n*n*sizeof(float));
	cudaMemcpy(d_a, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lower, lower[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_upper, upper[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

	cudaLUDecomp(a, n, d_a, d_lower, d_upper);

	
    //Get results from the device
    cudaMemcpy(lower[0],d_lower, n*n*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(upper[0],d_upper, n*n*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaThreadSynchronize();
	runtime = clock() - runtime; //Make note of LU Decomp


	// TODO: Write the substitution function. That is a future problem.
	
	printf("A:\n");
	PrintMatrix(a,n); 

	printf("Lower:\n");
	PrintMatrix(lower,n); 

	printf("Upper:\n");
	PrintMatrix(upper,n); 

	printf("LU Decomposition ran in %.2f seconds\n", (runtime)/float(CLOCKS_PER_SEC));
	
	cudaFree(d_lower);
	cudaFree(d_upper);
	cudaFree(d_a);

	DeleteMatrix(upper,n);	
	DeleteMatrix(lower,n);	
	DeleteMatrix(a,n);	

	return 0;
}

void sequentialLUdecomposition(float** a, float** &l, int n)
{
    for (int i = 0; i < n; i++)
    {
        
        float temp;
        float pivot = -1.0/a[i][i];
        
        l[i][i] = 1;
        
        for (int k = i+1; k < n; k++)
        {
            temp = pivot*a[k][i];
            l[k][i] = a[k][i]/a[i][i];
            for (int j = i; j < n; j++)
            {
                a[k][j] = a[k][j] + temp * a[i][j];
            }
        }
        
    }
}



//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& isPrint)
{
	bool isOK = true;
	
	if(argc < 2) 
	{
		printf("Arguments:<X> [<Y>]");
		printf("X : Matrix size [X x X]");
		printf("Y = 1: print the input/output matrix if X < 10");
		printf("Y <> 1 or missing: does not print the input/output matrix");
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			printf("Matrix size must be larger than 0");
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