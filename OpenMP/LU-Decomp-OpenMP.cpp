//-----------------------------------------------------------------------
// LU Decomposition : C++ OpenMP 
//-----------------------------------------------------------------------
//  Some features:
//   + Rowwise Data layout 
//  Programming by Anthony Dupont, Trystan Kaes, Tobby Lie, Marcus Gallegos
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
using namespace std;
//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& numThreads,int& isPrint)
{
	bool isOK = true;

	if(argc < 3) 
	{
		cout << "Arguments:<X> <Y> [<Z>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y : Number of threads" << endl;
		cout << "Z = 1: print the input/output matrix if X < 10" << endl;
		cout << "Z <> 1 or missing: does not print the input/output matrix" << endl;
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

		//get number of threads
		numThreads = atoi(argv[2]);
		if (numThreads <= 0)
		{	cout << "Number of threads must be larger than 0" <<endl;
			isOK = false;
		}

		//is print the input/output matrix
		if (argc >=4)
			isPrint = (atoi(argv[3])==1 && n <=9)?1:0;
		else
			isPrint = 0;
	}
	return isOK;
}

//------------------------------------------------------------------------------------------------
//Fills matrix A with random values, upper and lower is filled with 0's except for their diagonals
//------------------------------------------------------------------------------------------------
void InitializeMatrices(float a[][8], float lower[][8], float upper[][8], int size){
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
//Delete matrix matrix a[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **a,int n)
{
	delete[] a[0];
	delete[] a; 
}
//------------------------------------------------------------------
//Print the matrix that was passed to it
//------------------------------------------------------------------
void PrintMatrix(float matrix[][8], int size) 
{
	for (int i = 0 ; i < size ; i++){
		for (int j = 0 ; j < size ; j++){
			printf("%.2f\t", matrix[j][i]);
		}
		printf("\n");
	}
}
//------------------------------------------------------------------
//Compute the Gaussian Elimination for matrix a[n x n]
//------------------------------------------------------------------
bool ComputeGaussianElimination(float **a,int n)
{
	float pivot,gmax,pmax,temp;
	int  pindmax,gindmax,i,j,k;
	bool isOK;

	isOK = true;
	//Perform row wise elimination
	#pragma omp parallel shared(a, gmax, gindmax) firstprivate(n,k) private(pivot, i, j, temp, pmax, pindmax)
	for (k = 0 ; k < n-1 ; k++)
	{
		#pragma omp single
		{
			gmax = 0.0;	//only one thread needs to do this, the others can skip this since it's shared anyway
		}
		pmax = 0.0;

		//Find the pivot row among rows k, k+1,...n
		//Each thread works on a number of rows to find the local max value pmax
		//Then update this max local value to the global variable gmax
		//#pragma omp parallel shared(a,gmax,gindmax) firstprivate(n,k) private(pivot,i,j,temp,pmax,pindmax)
		//{	//parallel means spawning a bunch of threads
			//shared is all the shared memory variables
			//first private is a private variable that uses the previous value before entering parallel
			//private are private variables in each thread, not initialized
		//pmax = 0.0;
		#pragma omp for schedule(static) //dynamic scheduling method meaning that when it's done executing, it looks for another task to execute
										 //static would imply that it has x amount of iterations to execute and once it's done, it stops there
		//We search in the k column for the max value
		for (i = k ; i < n ; i++)
		{
			temp = abs(a[i][k]);     
		
			if (temp > pmax) 
			{
				pmax = temp;
				pindmax = i;
			}
		}
		#pragma omp critical
		{
		//this is a critical section
			if (gmax < pmax)
			{
				gmax = pmax;	//gmax is a shared variable across all threads
				gindmax = pindmax;
			}
		}
		//}
		
		//We want all threads to reach this point before moving forward
		#pragma omp barrier

		//If matrix is singular set the flag & quit
		if (gmax == 0){
			//return false;
			isOK = false;
			break;
			//#pragma omp cancel parallel		//can't return inside of a thread so we cancel the parallelization instead
		}

		//Swap rows if necessary
		if (gindmax != k)
		{
			#pragma omp for schedule(static)
			for (j = k; j < n; j++) 
			{	
				temp = a[gindmax][j];
				a[gindmax][j] = a[k][j];
				a[k][j] = temp;
			}
		}
			
		//Compute the pivot
		pivot = -1.0/a[k][k];
		// #pragma omp critical
		// {
			// //!!! FOR DEBUGGING PURPOSES ONLY, COMMENT THIS OUT WHEN RUNNING FOR ACCURATE TIME
			// cout << "k: " << k << ", Pivot: " << pivot << ", a[k][k]: " << a[k][k] << ", Thread: " << omp_get_thread_num() << endl;
			// PrintMatrix(a,n);
		// }
		
		//Perform row reductions
		#pragma omp for schedule(static)
		for(i = k+1; i < n; ++i){
			temp = pivot*a[i][k];	
			for(j = k; j < n; ++j){
				a[i][j] = a[i][j] + (temp*a[k][j]);	//a(k:n,i) = a(k:n,i) + temp*a(k:n,k)
			}
		}
		
	}

	return true;
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	srand(time(NULL));
	float a[8][8];
	float lower[8][8];
	float upper[8][8];
	
	int n,numThreads,isPrintMatrix;
	double runtime;
	bool isOK;
	
	if (GetUserInput(argc,argv,n,numThreads,isPrintMatrix)==false) return 1;

	//specify number of threads created in parallel region
	omp_set_num_threads(numThreads);

	//Initialize the value of matrix A, lower, and upper
	InitializeMatrices(a,lower,upper,n);
		
	if (isPrintMatrix) 
	{	
		printf("A:\n");
		PrintMatrix(a,n);
		printf("\nLower:\n");
		PrintMatrix(lower,n);
		printf("\nUpper:\n");
		PrintMatrix(upper,n);		
	}

	runtime = omp_get_wtime();
    
	//Compute LU Decomposition
	//isOK = ComputeGaussianElimination(a,n);

	runtime = omp_get_wtime() - runtime;

	/*if (isOK == true)
	{
		//The eliminated matrix is as below:
		if (isPrintMatrix)
		{
			cout<< "Output matrix:" << endl;
			PrintMatrix(a,n); 
		}

		//print computing time
		cout<< "Gaussian Elimination runs in "	<< setiosflags(ios::fixed) 
												<< setprecision(2)  
												<< runtime << " seconds\n";
	}
	else
	{
		cout<< "The matrix is singular" << endl;
	}*/
    
    // the code will run according to the number of threads specified in the arguments
    cout << "Matrix multiplication is computed using max of threads = "<< omp_get_max_threads() << " threads or cores" << endl;
    
    cout << " Matrix size  = " << n << endl;
    
	//DeleteMatrix(a,n);	
	return 0;
}
