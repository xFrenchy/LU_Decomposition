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


__global__ void RowOperations(float *lower, float *upper, int i, int thicness){

	// Let us get this diagonal thing out of the way
	if(blockIdx.x * blockDim.x  + threadIdx.x == 0) 
		lower[ i*thicness + i ] = 1; 

	int k = blockIdx.x + i + 1;
	int j = threadIdx.x + i;
	
	if( !( k < thicness && j < thicness) ) return; // Whoops

	__shared__ double pivot;

	// And get one pivot per block
	if(threadIdx.x == 0) 
		pivot = -1.0/upper[ i*thicness + i ];
	

	 // Hey guys! Wait up!
	__syncthreads();

	// It is worth noting that the matrices are column major here
	lower[k + thicness*i] = upper[k + thicness*i]/upper[i + thicness*i];
	upper[k + thicness*j] = upper[k + thicness*j] + pivot*upper[k + thicness*i] * upper[i + thicness*j];
	
    
}

void cudaLUDecomp(float *d_lower, float *d_upper, int thicness){

	int i, numBlocks, numThreads;

	for(i = 0; i < thicness; ++i){

		// Since all of these are square these are the same.
		numBlocks = numThreads = thicness-i;

		dim3 dimGrid(numBlocks,1);	
		dim3 dimBlock(numThreads,1);	

		RowOperations<<<dimGrid,dimBlock>>>(d_lower, d_upper, i, thicness);
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
	float *d_lower, *d_upper;
	
	int	n,isPrintMatrix;
	float runtime;

	//Get program input
	if (!GetUserInput(argc,argv,n,isPrintMatrix)) return 1;

	//Initialize the matrices
	// a == upper and lower -> 0
	InitializeMatrices(a, lower, upper, n);

	//Get start time
	runtime = clock()/(float)CLOCKS_PER_SEC;

	cudaMalloc((void**)&d_lower, n*n*sizeof(float));
	cudaMalloc((void**)&d_upper, n*n*sizeof(float));
	cudaMemcpy(d_upper, upper[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lower, lower[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

	cudaLUDecomp(d_lower, d_upper, n);

	
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
			upper[i][j] = a[i][j] = (rand() % 11) + 1;
			lower[i][j] = 0;
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