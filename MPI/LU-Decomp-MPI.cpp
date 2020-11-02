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


bool MatrixMultiplication(float **a, float *b, float *c, int n,int numProcesses, int myProcessID){
	c = new float[n];
	int nRows = n/numProcesses;		//we might have less processes than rows
	float *localb = new float[n*nRows];	//this is going to be a row
	float *localVectorb = new float[n];	//this is going to be the vector b
	int rank = 1;	//used to create a critical section without the master
	for(int i = 0; i < n; ++i){
		c[i] = 0.0;
	}
	MPI_Status status;
	
	printf("number of processes: %d\n", numProcesses);
	printf("myProcessID: %d\n", myProcessID);
	if(myProcessID == 0){
		//This for loop is used to send data to every single process that exists
		for(int k = numProcesses; k > 0; --k){
			//copy current row in matrix a into localb for workers to receive the data
			//printf("Sending this data:\n");
			for(int j = 0; j < nRows; ++j){
				int rowIndex = (k*nRows)-j - 1;
				//printf("Row Index: %d\n", rowIndex);
				for(int i = 0; i < n; ++i){
					localb[i+(n*j)] = a[rowIndex][i];	//Make sure that there isn't more processes than rows, k could be out of range
					//printf("%0.2f ", localb[i]);
				}
			}
			//printf("\n");
			//don't send data to master, master knows all, it would be insulting
			if(k != 0){
				MPI_Send(&localb[0],n*nRows,MPI_FLOAT,k-1,0,MPI_COMM_WORLD);	//send row of a
			}
		}
	}
	else{
		MPI_Recv(&localb[0],n*nRows,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
		// printf("Hey there! I received: \n");
		// for(int i = 0; i < n*nRows; ++i){
			// printf("%0.2f ", localb[i]);
		// }
		// printf("\n");
	}
	printf("\n");
	
	if(myProcessID == 0){
		for(int k = numProcesses - 1; k > 0; --k){
			for(int i = 0; i < n; ++i)
				localVectorb[i] = b[i];	//copy b to a local vector to distribute to workers
			MPI_Send(&localVectorb[0],n,MPI_FLOAT,k,0,MPI_COMM_WORLD);//send vector b
		}
	}
	else{
		MPI_Recv(&localVectorb[0],n,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
		// printf("Do I know b now???\n");
		// for(int i = 0; i < n; ++i){
			// printf("%0.2f ", localVectorb[i]);
		// }
	}
	
	//WOOO WE'VE SENT AND RECEIVED DATA CONGRATS now do the multiplication part like you're supposed to
	
	if(myProcessID > 0){
		for(int j = 0; j < nRows; ++j){
			localb[j] = localb[j*n] * localVectorb[0];
			for(int i = 1; i < n; ++i){
				localb[j] += localb[i+(j*n)] * localVectorb[i];
			}
		}
		//printf("I am row: %d\n", myProcessID);
		//printf("Row multiplication result: %0.2f \n", localb[0]);
	}
	else{
		//master gonna do some work too
		for(int j = 0; j < nRows; ++j){
			c[j] = a[j][0] * b[0];
			for(int i = 1; i < n; ++i){
				c[j] += a[j][i] * b[i];
			}
		}
	}
	//The result of the row multiplcation is stored in localb[0]. I can send localb[0] back and look specifically there
	
	if(myProcessID == 0){
		for (int k = 1 ; k < numProcesses ; k++) 
		{
			//receive data from worker process
			MPI_Recv(&localb[0],n,MPI_FLOAT,k,0,MPI_COMM_WORLD,&status);
			for(int j = 0; j < nRows; ++j){
				c[(k*nRows)+j] = localb[0];
			}
		}
	}
	else{
		MPI_Send(&localb[0],n,MPI_FLOAT,0,0,MPI_COMM_WORLD);
	}
	
	if(myProcessID == 0){
		printf("Output C Matrix: \n");
		for(int i = 0; i < n; ++i){
			printf("%0.2f\n", c[i]);
		}
	}
	return true;
}

//------------------------------------------------------------------
//Compute the Gaussian Elimination for matrix a[n x n]
//------------------------------------------------------------------
bool ComputeGaussianElimination(float **a,int n,int numProcesses, int myProcessID)
{
	float pivot,max,temp;
	int indmax,i,j,lk,k,master;
	int nCols = n/numProcesses;
	float *tmp = new float[n];
	float **b = new float*[nCols]; //create local matrix
	b[0] = new float[n*nCols];
	for (int j = 1; j < nCols; j++)
        b[j] = b[j-1] + n;
	MPI_Status status;
	//cout << "myProcessID" << myProcessID << endl; 				// this makes the difference. DONT KNOW WHY
	//process 0 send the data to compute nodes
	MPI_Scatter(a[0],n*nCols,MPI_FLOAT,b[0],n*nCols,MPI_FLOAT,0,MPI_COMM_WORLD);
	/* 
	 sendbuf
     Address of send buffer (choice, significant only at root). 
	 sendcount
     Number of elements sent to each process (integer, significant only at root). 
	 sendtype
     Datatype of send buffer elements (handle, significant only at root). 
	 recvcount
     Number of elements in receive buffer (integer). 
	 recvtype
     Datatype of receive buffer elements (handle). 
	 root
     Rank of sending process (integer). 
	 comm
     Communicator (handle). */

	//Perform rowwise multiplication
	for (k = 0 ; k < n ; k++)
	{
		//ID of master process
		master = k/nCols;
		
		//local k
		lk = k%nCols;

		//MPI_Bcast(&max,1,MPI_FLOAT,master,MPI_COMM_WORLD);
		//MPI_Bcast(&indmax,1,MPI_INT,master,MPI_COMM_WORLD);

		//Master 
		if (myProcessID == master)
		{
			pivot = -1.0/b[lk][k];
			for (i = k+1 ; i < n ; i++) 
				tmp[i]= pivot*b[lk][i];
		}

		MPI_Bcast(tmp + k + 1 ,n - k - 1,MPI_FLOAT,master,MPI_COMM_WORLD);

		//Perform row reductions
		if (myProcessID >= master)
		{
            for (j = ((myProcessID > master)?0:lk) ; j < nCols; j++)
			{
				for (i = k+1; i < n; i++)
				{
					b[j][i] = b[j][i] + tmp[i]*b[j][k];
				}
			}
		}
	}

	//process 0 collects results from the worker processes
	MPI_Gather(b[0],n*nCols,MPI_FLOAT,a[0],n*nCols,MPI_FLOAT,0,MPI_COMM_WORLD);

	delete[] b[0]; 
	delete[] b; 
	delete[] tmp; 
	return true;
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
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	srand(time(NULL));
	float a[8][8];
	float lower[8][8];
	float upper[8][8];
	
	int	n,isPrintMatrix,numProcesses,myProcessID;
	bool missing;
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
	//missing = MatrixMultiplication(a,b,c,n, numProcesses, myProcessID);

	//Master process gets end time and print results
	if (myProcessID == 0)
	{
		runtime = MPI_Wtime() - runtime;

		/*if (missing == true)
		{
			//Print result matrix
			if (isPrintMatrix==1)
			{
				printf("Output matrix:\n");
				//PrintMatrix(a,n); 
				//printMatrixB(c, n);	//this is actually printing c hehe horrible function title
			}
			printf("Gaussian Elimination runs in %.2f seconds \n", runtime);
 		}
		else
		{
			cout<< "The matrix is singular" << endl;
		}

		//All process delete matrix
		DeleteMatrix(a,n);	*/
	}

	MPI_Finalize();
	return 0;
}
