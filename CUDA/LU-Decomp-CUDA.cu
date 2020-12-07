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

bool GetUserInput(int, char *[], int&,int&);

void InitializeMatrices(float **&, float **&, float **&, int);
void InitializeVectors(float *&, float*&, float*&, int);

void PrintMatrix(float **, int);
void PrintVector(float*, int);

void DeleteMatrix(float**,int);
void DeleteVector(float *);

void LUDecomp(float *, float *, int);


__global__ void RowOperations(float *, float *, int, int);
__global__ void ForwardSubstitution(float *, float *, float* , int);
__global__ void BackwardSubstitution(float *, float *, float* , int);


//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]){
	srand(time(NULL));	//set the seed
	
	float **a, **lower, **upper; 				//Matrices
	float *b, *x, *y; 							//Vectors
	float *d_lower, *d_upper, *d_b, *d_x, *d_y; //Device pointers
	
	int	n,isPrint;
	float runtime;

	if (!GetUserInput(argc,argv,n,isPrint)) return 1;

	// a == upper and lower -> 0
	InitializeMatrices(a, lower, upper, n);
	InitializeVectors(x, y, b, n);

	//Get start time
	runtime = clock()/(float)CLOCKS_PER_SEC;

	// ######################### BEGIN LU Decomp ##############################3
		cudaMalloc((void**)&d_lower, n*n*sizeof(float));
		cudaMalloc((void**)&d_upper, n*n*sizeof(float));
		cudaMemcpy(d_upper, upper[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_lower, lower[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

		LUDecomp(d_lower, d_upper, n);

		cudaDeviceSynchronize();
	// ######################### END LU Decomp ##############################3


	// ######################### BEGIN Substitution ##############################
		cudaMalloc((void**)&d_b, n*sizeof(float));
		cudaMalloc((void**)&d_x, n*sizeof(float));
		cudaMalloc((void**)&d_y, n*sizeof(float));
		cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

		dim3 dimGrid(n,1);	
		dim3 dimBlock(n,1);

		ForwardSubstitution<<<dimGrid, dimBlock>>>(d_lower, d_y, d_b, n);
		BackwardSubstitution<<<dimGrid, dimBlock>>>(d_upper, d_x, d_y, n);
		
		cudaThreadSynchronize();
	// ######################### END Substitution ##############################3


	// ######################### BEGIN Copy Back ##############################
		cudaMemcpy(lower[0],d_lower, n*n*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(upper[0],d_upper, n*n*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(x,d_x, n*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(y,d_y, n*sizeof(float),cudaMemcpyDeviceToHost);
	// ######################### END Copy Back ##############################

	runtime = clock() - runtime; //Make note of time

	if(isPrint == 1){
		printf("A:\n");
		PrintMatrix(a,n); 

		printf("B:\n");
		PrintVector(b,n); 

		printf("--------------------------------------------------\n");

		printf("Lower:\n");
		PrintMatrix(lower,n); 

		printf("Upper:\n");
		PrintMatrix(upper,n); 

		printf("Y:\n");
		PrintVector(y,n); 

		printf("X:\n");
		PrintVector(x,n); 

	}

	printf("LU Decomposition and Forward/Backward substitution to solve Ax=B ran in %.2f seconds\n", (runtime)/float(CLOCKS_PER_SEC));
	
	
	cudaFree(d_lower);
	cudaFree(d_upper);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_y);

	DeleteMatrix(upper,n);	
	DeleteMatrix(lower,n);	
	DeleteMatrix(a,n);	

	DeleteVector(x);
	DeleteVector(y);
	DeleteVector(b);

	return 0;
}

//------------------------------------------------------------------
//	KERNEL DRIVERS
//------------------------------------------------------------------


void LUDecomp(float *d_lower, float *d_upper, int thicness){

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
//	KERNELS
//------------------------------------------------------------------
__global__ void RowOperations(float *lower, float *upper, int i, int thicness){

	// Let us get this diagonal thing out of the way
	if(blockIdx.x * blockDim.x  + threadIdx.x == 0) 
		lower[ i*thicness + i ] = 1; 

	int k = blockIdx.x + i + 1;
	int j = threadIdx.x + i;
	
	if( !( k < thicness && j < thicness) ) return; // Whoops

	__shared__ float pivot;

	// And get one pivot per block
	if(threadIdx.x == 0) 
		pivot = -1.0/upper[ i*thicness + i ];
	

	 // Hey guys! Wait up!
	__syncthreads();

	// It is worth noting that the matrices are column major here
	lower[k + thicness*i] = upper[k + thicness*i]/upper[i + thicness*i];
	upper[k + thicness*j] = upper[k + thicness*j] + pivot*upper[k + thicness*i] * upper[i + thicness*j];
    
}

__global__ void ForwardSubstitution(float *lower, float *y, float* b, int thicness){
	if(blockIdx.x * blockDim.x  + threadIdx.x == 0) // Last Element
		y[0] = b[0] / lower[0];

	int i = blockIdx.x + 1; 
	int j = i - threadIdx.x;

	if( !( i < thicness && j < thicness) || ( j < 0 )  ) return; // Whoops

	__shared__ float temp;

	if(threadIdx.x == 0) 
		temp = b[i];

	// Hey guys! Wait up!
	__syncthreads();

	temp = temp - lower[ i + thicness*j] * y[j];

	// Hey guys! Wait up!
	__syncthreads();

	y[i] = temp/lower[i + thicness*i];
}

__global__ void BackwardSubstitution(float *upper, float *x, float* y, int thicness){
	if(blockIdx.x * blockDim.x  + threadIdx.x == 0) 
		x[thicness - 1] = y[thicness - 1] / upper[(thicness - 1) + thicness*(thicness-1)]; // Last Element

	int i = thicness - blockIdx.x - 2;
	int j = thicness - i - threadIdx.x - 1;

	if( !( i < thicness && j < thicness) || ( j < 0 )  ) return; // Whoops

	__shared__ float temp;

	if(threadIdx.x == 0) 
		temp = y[i];

	// Hey guys! Wait up!
	__syncthreads();

	temp = temp - upper[ i + thicness*j] * x[j];

	// Hey guys! Wait up!
	__syncthreads();

	x[i] = temp/upper[i + thicness*i];
}



//------------------------------------------------------------------
//	UTILITIES
//------------------------------------------------------------------

//   Get user input of matrix dimension and printing option
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



//delete matrix matrix a[n x n]
void DeleteMatrix(float **a,int n)
{
	delete[] a[0];
	delete[] a; 
}


void DeleteVector(float* x)
{
    delete[] x;
}

//Fills matrix A with random values, upper and lower is filled with 0's except for their diagonals
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


void InitializeVectors(float *&x, float *&y, float*& b, int n) {

    // allocate square 2d matrix
	x = new float[n];
	y = new float[n];
	b = new float[n];
    
    
    for (int j = 0 ; j < n ; j++) {
		b[j] = (float)(rand() % 11) + 1;
		x[j] = y[j] = 0;
    }
}


//Print the matrix that was passed to it
void PrintMatrix(float **matrix, int size) 
{
	for (int i = 0 ; i < size ; i++){
		for (int j = 0 ; j < size ; j++){
			printf("%.2f\t", matrix[j][i]);
		}
		printf("\n");
	}
}

void PrintVector(float* x, int n)
{
    for (int j = 0 ; j < n ; j++) {
        printf("%.2f\n", x[j]);
    }
}