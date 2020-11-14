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

//------------------------------------------------------------------------------------------------
//Fills matrix A with random values
//------------------------------------------------------------------------------------------------
void InitializeMatrix(float **&a, int size){
	a = new float*[size];
	a[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		a[i] = a[i-1] + size;
}


//------------------------------------------------------------------------------------------------
//Fills vector with random values
//------------------------------------------------------------------------------------------------
float* InitializeVector(float *vector, int size){
	vector = new float[size];
	for(int i = 0; i < size; ++i){
		vector[i] = float(i+1)/size;
		//printf("%.2f\t", b[i]);
	}
	return vector;
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


void PrintVector(float *vector, int size){
	for(int i = 0; i < size; ++i){
			printf("%0.2f, ", vector[i]);
	}
	printf("\n");
}


//Assumes a is column major, stores row major in b
void ColumnMajorToRowMajor(float **&a, float **&b, int size){
	b = new float*[size];
	b[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		b[i] = b[i-1] + size;
	
	for(int i = 0; i < size; ++i){
		for(int j = 0; j < size; ++j){
			b[i][j] = a[j][i];
		}
	}
}


void LUDecomp1(float **a, float **lower, float **upper, int size, int numProcesses, int myProcessID){
	/* This is the broken version. It doesn't really do LU Decomposition. I was doing it wrong this whole time
	 * Send each worker their own unique row
	 * Worker performs LU Decomposition on their row
	 * Worker sends updated data back to master
	 * Master places data into Lower and Upper matrix
	 * Keep sending rows to all workers until all rows have been worked on
	 */
	MPI_Status status;
	int rowSize = size/numProcesses;	//amount of rows that a worker will work on. Example if Matrix size if 8 and we have 4 processes, each worker will have 2 rows
	//Need to send each row of A to each worker, therefore I need a local 1d array since they only only receive a row

	//localA has the number of pivots needed for this specific row in [0], followed by the first row of matrix A, followed by the row sent by the master
	float localA[1 + size*2];	//[1, 7,11,7,3,2, 5,1,7,4,2]	1 pivot, 7,11,7... first row in matrix A, 5,1,7... row sent by master
	//Workers will need to find their pivot based off the first row of the matrix, so localA will always have the first row
	if(myProcessID == 0){
		for(int i = 0; i < size; ++i){
			localA[i+1] = a[0][i];
			upper[0][i] = a[0][i];
		}
	}
	int rowNumber = 0;
	for(int j = 0; j < rowSize; ++j){
		if(myProcessID == 0){
			for(int k = 1; k < numProcesses; ++k){
				++rowNumber;
				//copy current row in matrix a into localA for workers to receive the data
				//printf("Sending this data:\n");
				localA[0] = float(rowNumber);	//The worker will know how many times to loop with this number
				for(int i = 0; i < size; ++i){
					localA[(size+1)+i] = a[rowNumber][i];	//place data into localA
					//printf("%0.2f ", localA[size+1+i]);
				}
				//printf("I sent: %d\n", rowNumber);
				
				
				//don't send data to master, master knows all, it would be insulting
				MPI_Send(&localA[0],(1 + size*2),MPI_FLOAT,k,0,MPI_COMM_WORLD);	//send row(s) of a
			}
			++rowNumber;
			//Master won't just stand back and do nothing! It's a process too! Get to work mister master >:(
			//Compute LU Decomposition on the row I need a pivot
			float pivots[size];	//depending on which row we perform LU Decomposition on, there could be many pivots, they will all be stored here
			
			//the first row will have 1 entry, 2nd row will have 2, 3rd row will have 3, etc. So I will use the rowNumber passed to the worker
			int index = j+((numProcesses-1)*j);
			//printf("Row: %d\n", index);
				for(int i = 0; i < index; ++i){
					pivots[i] = (a[index][i]/a[0][i]);
					//printf("top: %0.2f , bottom: %0.2f", a[index][i], a[0][i]);
					//printf("Pivot: %0.2f\n", pivots[i]);
					lower[index][i] = pivots[i];
					//This pivot will go into Lower, apply pivot on row (first element should go compute to 0), final updated row will go in Upper
					for(int h = i; h < size; ++h){
						upper[index][h] = a[index][h] - (a[0][h]*pivots[i]);	//applying pivot to the row
					}
				}
			/*
			for(int n = 0; n <= j; ++n){
				lower[j+((numProcesses-1)*j)][n] = 0;
			}
			for(int p = 0; p <= j; ++p){
				upper[j+((numProcesses-1)*j)][p] = 0;
			}*/
			
			//Workers received their data, now master does a receive and it should be a blocking statement so it waits for data
			for(int m = 1; m < numProcesses; ++m){
				MPI_Recv(&localA[0],(1 + size*2),MPI_FLOAT,m,0,MPI_COMM_WORLD,&status);
				//construct the upper and lower based on what we received
				int index = int(localA[0]);
				for(int n = 0; n < index; ++n){
					lower[index][n] = localA[1+n];
				}
				for(int p = 0; p < size; ++p){
					upper[index][p] = localA[(size+1)+p];
				}
			}
			for(int n = 0; n < size; ++n){
				localA[n+1] = a[0][n];	//I'm dumb dumb and because the workers overwrite this data, it messed up the data when I go back up the for loop for the next iteration
			}
		}
		else{
			MPI_Recv(&localA[0],(1 + size*2),MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
			//DEBUG PURPOSE ONLY
			/*if(myProcessID == 1){
				printf("Hey there! I received: \n");
				for(int i = 0; i < 1+size*2; ++i){
					printf("%0.2f ", localA[i]);
				}
			}
			printf("\n");*/
			//Compute LU Decomposition on the row received then send it back. I need a pivot
			float pivots[size];	//depending on which row we perform LU Decomposition on, there could be many pivots, they will all be stored here
			
			//the first row will have 1 entry, 2nd row will have 2, 3rd row will have 3, etc. So I will use the rowNumber passed to the worker
			int index = int(localA[0]);
				for(int i = 0; i < index; ++i){
					pivots[i] = (localA[(size+1)+i]/localA[1+i]);	//localA[size+1] is the beginning of the first row sent from the data partition 
																//since localA[1-size] is the first row of the matrix A
					//This pivot will go into Lower, apply pivot on row (first element should go compute to 0), final updated row will go in Upper
					for(int j = i; j < size; ++j){
						localA[size+j+1] = localA[(size+1)+j] - (localA[1+j]*pivots[i]);	//applying pivot to the row
					}
				}
			
			//set up the array to send back the pivots for L, and the final updated row for U
			for(int i = 0; i < index; ++i){
				localA[1+i] = pivots[i];	//overriding the beginning data of the first row of Matrix A since that isn't needed anymore
			}
			
			//DEBUG PURPOSE ONLY
			/*if(myProcessID == 1){
				printf("My pivots and final row: \n");
				for(int i = 0; i < 1+size*2; ++i){
					printf("%0.2f ", localA[i]);
					if(i == 0)
						printf(" : ");
					if(i == size+1)
						printf(" : ");
				}
			}
			printf("\n");*/
			MPI_Send(&localA[0],1 + size*2,MPI_FLOAT,0,0,MPI_COMM_WORLD);
		}
	}
}


void LUDecomp(float **a, float **lower, float **upper, int size, int numProcesses, int myProcessID){
	/* Send each worker their own unique row
	 * Worker performs LU Decomposition on their row
	 * Worker sends updated data back to master
	 * Master places data into Lower and Upper matrix
	 * Keep sending rows to all workers until all rows have been worked on
	 */
	MPI_Status status;
	int rowSize = size/numProcesses;	//amount of rows that a worker will work on. Example if Matrix size if 8 and we have 4 processes, each worker will have 2 rows
	int master, lk, indmax;
	float pivot, max, temp;
	float *tmp = new float[size];
	float **b = new float*[rowSize]; //create local matrix
	b[0] = new float[size*rowSize];
	for (int j = 1; j < rowSize; j++)
        b[j] = b[j-1] + size;
	float **c = new float*[rowSize]; //create local matrix
	c[0] = new float[size*rowSize];
	for (int j = 1; j < rowSize; j++)
        c[j] = c[j-1] + size;
	MPI_Scatter(a[0],size*rowSize,MPI_FLOAT,b[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Scatter(lower[0],size*rowSize,MPI_FLOAT,c[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	//cout << "hi\n";
	for (int k = 0 ; k < size ; ++k)
	{
		max = 0.0;
		indmax = k;
		//printf("process %d has max %f\n",myProcessID, max );

		//ID of master process
		master = k/rowSize;
		
		//local k
		lk = k%rowSize;

		//Only master process find the pivot row
		//Then broadcast it to all other processes

		if (myProcessID == master)
		{	
			//Find the pivot row
			for (int i = k ; i < size ; i++) 
			{	
				temp = abs(b[lk][i]);     
				if (temp > max) 
				{
				  max = temp;
				  indmax = i;
				}
			}
		}

		MPI_Bcast(&max,1,MPI_FLOAT,master,MPI_COMM_WORLD);
		MPI_Bcast(&indmax,1,MPI_INT,master,MPI_COMM_WORLD);


		//If matrix is singular set the flag & quit
		if (max == 0) return;

		//Master 
		if (myProcessID == master)
		{
			pivot = -1.0/b[lk][k];
			//c[lk][k] = pivot;
			//cout << pivot << endl;
			for (int i = k+1 ; i < size ; i++){ 
				tmp[i]= pivot*b[lk][i];
				c[lk][i] = -1*(pivot*b[lk][i]);
				//cout << tmp[i] << " ";
			}
		}

		MPI_Bcast(tmp + k + 1 ,size - k - 1,MPI_FLOAT,master,MPI_COMM_WORLD);

		//Perform row reductions
		if (myProcessID >= master)
		{
            for (int j = ((myProcessID > master)?0:lk) ; j < rowSize; j++)
			{
				for (int i = k+1; i < size; i++)
				{
					b[j][i] = b[j][i] + tmp[i]*b[j][k];
				}
			}
		}
	}
	
	MPI_Gather(b[0],size*rowSize,MPI_FLOAT,a[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(c[0],size*rowSize,MPI_FLOAT,lower[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
}


void forwardSubstitution(float **lower, float *vector, int size, int numProcesses, int myProcessID){
	//So now that we have the LU matrix, it's time to do something with it I guess
	//We'll work from the top to bottom for forward. Solve row 1, broadcast the value to all other for them to plug it in
	//I think master should always be solving the current row and broadcasting to others
	MPI_Status status;
	int rowSize = size/numProcesses;	//amount of rows that a worker will work on. Example if Matrix size if 8 and we have 4 processes, each worker will have 2 rows
	int master;
	float bcastVal = 0.00;
	float *tmp = new float[size];	//will broadcast the vector in here for workers
	float **b = new float*[rowSize]; //create local matrix
	b[0] = new float[size*rowSize];
	for (int j = 1; j < rowSize; j++){
		b[j] = b[j-1] + size;
	}
	float *solution = new float[size];
	float **lowerRow;
	lowerRow = new float*[size];
	lowerRow[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		lowerRow[i] = lowerRow[i-1] + size;
	
	//Set up memory for master to send to workers
	if(myProcessID == 0){
		for(int i = 0; i < size; ++i){
			tmp[i] = vector[i];
		}
		//convert from column major to row major
		for(int i = 0; i < size; ++i){
			for(int j = 0; j < size; ++j){
				lowerRow[i][j] = lower[j][i];
			}
		}
		//DEBUGGING
		/*for(int i = 0; i < size; ++i)
			for(int j = 0; j < size; ++j)
				printf("%0.2f, ", lowerRow[i][j]);
		//printf("%0.2f\n",lowerRow[0]);*/
	}
	
	/*if(myProcessID == 1){
			for(int j = 0; j < rowSize; ++j){
				for(int k = 0; k < size; ++k){
					printf("%0.2f, ", b[j][k]);
				}
			}
			printf("End\n");
		}*/
	MPI_Bcast(tmp,size,MPI_FLOAT,0,MPI_COMM_WORLD);
	//printf("Size*rowSize: %d\n", size*rowSize);
	//printf("%0.2f\n", b[0]);
	MPI_Scatter(lowerRow[0],size*rowSize,MPI_FLOAT,b[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	for(int i = 0; i < size; ++i){
		//Iterate over the length of a row chunk
		
		master = i/rowSize;	
		//Solve one step of each row in a parallel fashion 'snap' before iterating over to the index
		if(myProcessID == master){
			//This row is solvable, solve it, broadcast it
			bcastVal = (tmp[i]/b[i%rowSize][i]);
			solution[i] = bcastVal;
			printf("x%d: %0.2f\n", i, bcastVal);
			
			for(int j = 1; j < rowSize; ++j){
				tmp[j] -= b[j][i]*bcastVal;
			}
		}
			
		//BEEEEEEEEEEEEEEEEP BEEEEEEEEEEEEEEEEEP BEEEEEEEEEEEEEEEP WE INTERUPT THIS PROGRAM BECAUSE SOME X WAS SOLVED
		MPI_Bcast(&bcastVal, 1, MPI_FLOAT, master, MPI_COMM_WORLD);
		
		for(int j = 1; j <= rowSize; ++j){
			if(myProcessID > master){
				tmp[myProcessID*rowSize + j] -= b[j-1][i]*bcastVal;
			}
			//DEBUGGING PRINT
			/*if(myProcessID == 2){
				for(int j = 0; j < rowSize; ++j){
					for(int k = 0; k < size; ++k){
						printf("%0.2f, ", b[j][k]);
						//printf("%0.2f, ", tmp[k]);
					}
				}
				printf("End\n");
			}*/
		}
	}
	MPI_Gather(b[0],size*rowSize,MPI_FLOAT,lowerRow[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	//Let's recv the solution matrix
	if(myProcessID == 0){
		for(int j = 0; j < rowSize; ++j){
			vector[j] = solution[j];
			printf("%0.2f, ", vector[j]);
		}
		for (int k = 1 ; k < numProcesses ; k++) 
		{
			//receive data from worker process
			MPI_Recv(tmp,rowSize,MPI_FLOAT,k,0,MPI_COMM_WORLD,&status);
			for(int m = 0; m < rowSize; ++m){
				vector[(k*rowSize)+m] = tmp[m];
				printf("%0.2f, ", vector[(k*rowSize)+m]);
			}
		}
	}
	else{
		MPI_Send(&solution[myProcessID*rowSize],rowSize,MPI_FLOAT,0,0,MPI_COMM_WORLD);
	}
}


void backwardSubstitution(float **upper, float *vector, int size, int numProcesses, int myProcessID){
	//We've gone forward, but like, why not go backward as well you know?
	//Started from the bottom now we here, ay, ya, ay
	//I think master should always be solving the current row and broadcasting to others
	MPI_Status status;
	int rowSize = size/numProcesses;	//amount of rows that a worker will work on. Example if Matrix size if 8 and we have 4 processes, each worker will have 2 rows
	int master;
	float bcastVal = 0.00;
	float *tmp = new float[size];	//will broadcast the vector in here for workers
	float **b = new float*[rowSize]; //create local matrix
	b[0] = new float[size*rowSize];
	for (int j = 1; j < rowSize; j++){
		b[j] = b[j-1] + size;
	}
	float *solution = new float[size];
	float **upperRow;
	upperRow = new float*[size];
	upperRow[0] = new float[size*size];
	for (int i = 1; i < size; i++)	
		upperRow[i] = upperRow[i-1] + size;
	
	//Set up memory for master to send to workers
	if(myProcessID == 0){
		for(int i = 0; i < size; ++i){
			tmp[i] = vector[i];
		}
		//convert from column major to row major
		for(int i = 0; i < size; ++i){
			for(int j = 0; j < size; ++j){
				upperRow[i][j] = upper[j][i];
			}
		}
		//DEBUGGING
		/*for(int i = 0; i < size; ++i)
			for(int j = 0; j < size; ++j)
				printf("%0.2f, ", upperRow[i][j]);
		//printf("%0.2f\n",lowerRow[0]);*/
	}
	
	/*if(myProcessID == 1){
			for(int j = 0; j < rowSize; ++j){
				for(int k = 0; k < size; ++k){
					printf("%0.2f, ", b[j][k]);
				}
			}
			printf("End\n");
		}*/
	MPI_Bcast(tmp,size,MPI_FLOAT,0,MPI_COMM_WORLD);
	//printf("Size*rowSize: %d\n", size*rowSize);
	//printf("%0.2f\n", b[0]);
	MPI_Scatter(upperRow[0],size*rowSize,MPI_FLOAT,b[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
	for(int i = size-1; i >= 0; --i){
		//Iterate over the length of a row chunk
		master = i/rowSize;
		//Solve one step of each row in a parallel fashion 'snap' before iterating over to the index
		if(myProcessID == master){
			//This row is solvable, solve it, broadcast it
			bcastVal = (tmp[i]/b[i%rowSize][i]);
			//printf("I am looking at this: %0.2f", b[0][i]);
			solution[i] = bcastVal;
			printf("y%d: %0.2f\n", i, bcastVal);
			for(int j = 1; j < rowSize; ++j){
				tmp[j] -= b[j][i]*bcastVal;
			}
		}
		
		//BEEEEEEEEEEEEEEEEP BEEEEEEEEEEEEEEEEEP BEEEEEEEEEEEEEEEP WE INTERUPT THIS PROGRAM BECAUSE SOME X WAS SOLVED
		MPI_Bcast(&bcastVal, 1, MPI_FLOAT, master, MPI_COMM_WORLD);
		
		for(int j = 1; j <= rowSize; ++j){
			if(myProcessID < master){
				//printf("b[0][i] is: %0.2f\n", b[0][i]);
				tmp[myProcessID*rowSize + j] -= b[j-1][i]*bcastVal;
			}
		}
		//DEBUGGING PRINT
		/*if(myProcessID == 1){
			for(int j = 0; j < rowSize; ++j){
				for(int k = 0; k < size; ++k){
					printf("%0.2f, ", b[j][k]);
					//printf("%0.2f, ", tmp[k]);
				}
			}
			printf("End\n");
		}*/
	}
	MPI_Gather(b[0],size*rowSize,MPI_FLOAT,upperRow[0],size*rowSize,MPI_FLOAT,0,MPI_COMM_WORLD);
}

//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[]){
	srand(time(NULL));	//set the seed
	
	//Matrices
	float **a;
	float **lower;
	float **upper;
	float *vector;
	//float **lowerRow;
	
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
		vector = InitializeVector(vector, n);
		
		printf("A:\n");
		PrintMatrix(a,n); 
		printf("\nSolution vector: \n");
		PrintVector(vector, n);
		

		//Get start time
		runtime = MPI_Wtime();
	}
	
	//Compute the LU Decomposition
	LUDecomp(a, lower, upper, n, numProcesses, myProcessID);

	
	//if (myProcessID == 0){
		//ColumnMajorToRowMajor(lower, lowerRow, n);
		//MPI_Bcast(lowerRow, n*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
		// printf("\nLower:\n");
		// PrintMatrix(lower,n);
		// printf("\nLowerRow:\n");
		// PrintMatrix(lowerRow,n);
	//}
	if(myProcessID == 0){
		printf("Forward Substitution:\n");
	}
	
	forwardSubstitution(lower, vector, n, numProcesses, myProcessID);
	
	if(myProcessID == 0){
		printf("\nBackward Substitution\n");
	}
	//Vector has been updated from forwardSub, so it's using the solved solution vector for backward sub
	backwardSubstitution(a, vector, n, numProcesses, myProcessID);
	
	//cout << "Hyyiiaaa?" << endl;
	//Master process gets end time and print results
	if (myProcessID == 0)
	{
		runtime = MPI_Wtime() - runtime;
		
		printf("\nResults:\n");
		printf("\nLower:\n");
		PrintMatrix(lower,n);
		printf("\nUpper:\n");
		PrintMatrix(a,n);

		//All process delete matrix, ahhaa, yeah, maybe when I get that to work one day
		DeleteMatrix(a,n);
		DeleteMatrix(lower, n);
		DeleteMatrix(upper, n);
		//DeleteMatrix(lowerRow, n);
	}

	MPI_Finalize();
	return 0;
}
