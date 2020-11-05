//-----------------------------------------------------------------------
// Parallel LU Decomposition - C++ MPI
//-----------------------------------------------------------------------
//  Some features:
//  + For simplicity, program requires matrix size must be multiple 
//    of processes. 
//  + Data partition model of each process is block
//  + Data communication: Send / Recv
//  Programming by Anthony Dupont, Trystan Kaes, Tobby Lie, Marcus Gallegos 
//  Update in 11/1/2020
// ----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>    
using namespace std;


//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& isPrint,int numProcesses, int myProcessID)
{
	bool isOK = true;
	
	if(argc < 2) 
	{
		if (myProcessID==0) 
		{
			cout << "Arguments:<X> [<Y>]" << endl;
			cout << "X : Matrix size [X x X]" << endl;
			cout << "Y = 1: print the input/output matrix if X < 10" << endl;
			cout << "Y <> 1 or missing: does not print the input/output matrix" << endl;
		}
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			if (myProcessID==0) cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}
		//check if matrix size is multiple of processes
		if ( ( n % numProcesses ) != 0 )
		{
			if (myProcessID==0) cout << "Matrix size must be multiple of the number of processes" <<endl;
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
void InitializeMatrices(float a[][5], float lower[][5], float upper[][5], int size){
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
void PrintMatrix(float matrix[][5], int size) 
{
	for (int i = 0 ; i < size ; i++){
		for (int j = 0 ; j < size ; j++){
			printf("%.2f\t", matrix[i][j]);
		}
		printf("\n");
	}
}


void LUDecomp(float a[][5], float lower[][5], float upper[][5], int size, int numProcesses, int myProcessID){
	MPI_Status status;
	//Need to send each row of A to each worker, therefore I need a local 1d array since they only only receive a row
	float locala[size*2];	//locala has defaulty the first row + the needed row(s) the worker has to receive
	
	//Workers will need to find their pivot based off the first row of the matrix, so locala will always have the first row
	for(int i = 0; i < size; ++i){
		locala[i] = a[0][i];
	}
	
	//Copy data from Matrix A into local, and send each row to different workers
	if(myProcessID == 0){
		//This for loop is used to send data to every single process that exists
		for(int k = numProcesses; k > 0; --k){
			//copy current row in matrix a into locala for workers to receive the data
			//printf("Sending this data:\n");
			//for(int j = 0; j < nRows; ++j){
			//	int rowIndex = (k*nRows)-j - 1;
				//printf("Row Index: %d\n", rowIndex);
				for(int i = 0; i < size; ++i){
					locala[i+size] = a[k-1][i];	//Make sure that there isn't more processes than rows, k could be out of range
					//printf("%0.2f ", locala[i]);
				}
			//}
			//printf("\n");
			//don't send data to master, master knows all, it would be insulting
			MPI_Send(&locala[0],size*2,MPI_FLOAT,k-1,0,MPI_COMM_WORLD);	//send row of a
		}
	}
	else{
		MPI_Recv(&locala[0],size*2,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
		/*printf("Hey there! I received: \n");
		for(int i = 0; i < size*2; ++i){
			printf("%0.2f ", locala[i]);
		}
		printf("\n");*/
	}
	printf("\n");
	
	//Workers have their rows, now perform LU Decomp on that specific row. I need a pivot
	if(myProcessID > 0){
		float pivots[size];
		//the first row will have 1 entry, 2nd row will have 2, 3rd row will have 3, etc. So I will use their index to know how many entries they have
		for(int i = 0; i < myProcessID; ++i){
			pivots[i] = (locala[size+i]/locala[0+i]);	//locala[size] is the beginning of the first row sent since locala[0-size] is the first row of the matrix
			//This pivot will go into Lower, apply pivot on row (first element should go to 0), final updated row will go in Upper
			for(int j = i; j < size; ++j){
				locala[size+j] = (locala[0+j]*pivots[i]) - locala[size+j];
			}
		}
		//set up the array to send back the pivots for L, and the final updated row for U
		for(int i = 0; i < myProcessID; ++i){
			locala[i] = pivots[i];	//overriding the very fist row of Matrix A, don't need that data anymore
		}
		
		//DEBUG PURPOSE ONLY
		/*printf("My pivots and final row: \n");
		for(int i = 0; i < size*2; ++i){
			printf("%0.2f ", locala[i]);
			if(i == size)
				printf(" : ");
		}
		printf("\n");*/
	}
	
	if(myProcessID == 0){
		//first, master will set the row for upper equal to the first row of A
		for(int i = 0; i < size; ++i){
			upper[0][i] = a[0][i];
		}
		//Master will receive data from workers to create the rest of upper and lower
		for(int i = 1; i < numProcesses; ++i){
			MPI_Recv(&locala[0],size*2,MPI_FLOAT,i,0,MPI_COMM_WORLD,&status);
			for(int j = 0; j < i; ++j){
				lower[i][j] = locala[j];	//the beginning of locala should be the pivots, everything past that is the row
			}
			for(int j = 0; j < size; ++j){
				upper[i][j] = locala[j+size];
			}
		}
	}
	else{
		//Workers send data to master
		MPI_Send(&locala[0],size*2,MPI_FLOAT,0,0,MPI_COMM_WORLD);
	}
}


//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]){
	srand(time(NULL));	//set the seed
	
	//Matrices
	float a[5][5];
	float lower[5][5];
	float upper[5][5];
	
	int	n,isPrintMatrix,numProcesses,myProcessID;
	double runtime;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
	MPI_Comm_rank(MPI_COMM_WORLD, &myProcessID);
	
	//Get program input
	if (GetUserInput(argc,argv,n,isPrintMatrix,numProcesses,myProcessID)==false)
	{
		MPI_Finalize();
		return 1;
	}

	//Master part
	if (myProcessID == 0) 
	{	
		//Initialize the matrices
		InitializeMatrices(a, lower, upper, n);

		//Prints the input maxtrix if needed
		if (isPrintMatrix==1)
		{
			printf("A:\n");
			PrintMatrix(a,n); 
			printf("\nLower:\n");
			PrintMatrix(lower,n);
			printf("\nUpper:\n");
			PrintMatrix(upper,n);
		}

		//Get start time
		runtime = MPI_Wtime();
	}

	//Compute the LU Decomposition
	LUDecomp(a, lower, upper, n, numProcesses, myProcessID);
	
	//Master process gets end time and print results
	if (myProcessID == 0)
	{
		runtime = MPI_Wtime() - runtime;
		
		printf("\nResults:\n");
		printf("\nLower:\n");
		PrintMatrix(lower,n);
		printf("\nUpper:\n");
		PrintMatrix(upper,n);

		//All process delete matrix
		DeleteMatrix(a,n);	*/
	}

	MPI_Finalize();
	return 0;
}
