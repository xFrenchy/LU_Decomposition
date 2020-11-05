//-----------------------------------------------------------------------
// LU Decomp - Sequential
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <cstdlib>
using namespace std;
//-----------------------------------------------------------------------
//   Get user input for matrix dimension or printing option
//-----------------------------------------------------------------------

typedef float** twoDPtr; 

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

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
float** InitializeMatrix(int n, float value)
{

	// allocate square 2d matrix
	float **x = new float*[n];
	for(int i = 0 ; i < n ; i++)
		x[i] = new float[n] ;


	// assign random values
    srand (time(NULL));
	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{
            if (value == 1)  // generate input matrices (a and b)
                x[i][j] = (float)((rand()%10 + 1)/(float)2);
            else
                x[i][j] = 0;  // initializing resulting matrix
		}
	}

	return x ;
}
//------------------------------------------------------------------
//Delete matrix x[n x n]
//------------------------------------------------------------------
void DeleteMatrix(float **x,int n)
{

	for(int i = 0; i < n ; i++)
		delete[] x[i];
	
}
//------------------------------------------------------------------
//Print matrix	
//------------------------------------------------------------------
void PrintMatrix(float** x, int n) 
{

	for (int i = 0 ; i < n ; i++)
	{
		cout<< "Row " << (i+1) << ":\t" ;
		for (int j = 0 ; j < n ; j++)
		{
			cout << setiosflags(ios::fixed) << setprecision(2) << x[i][j] << " ";
		}
		cout << endl ;
	}
}
//------------------------------------------------------------------
//Do LU Decomp
//------------------------------------------------------------------
void LUdecomposition(float** a, float** &l, int n)
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
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	int	n,isPrint;
	double runtime;

	if (GetUserInput(argc,argv,n,isPrint)==false) return 1;

    cout << "Starting sequential LU decomposition" << endl;
    cout << "matrix size = " << n << "x " << n << endl;

	//Initialize the value of matrix a, l, u
	float **a = InitializeMatrix(n, 1.0);
	float **l = InitializeMatrix(n, 0.0);

	//Print the input matrices
	if (isPrint==1)
	{
		cout<< "Matrix A:" << endl;
		PrintMatrix(a,n);
	}

	runtime = clock()/(double)CLOCKS_PER_SEC;

    LUdecomposition(a,l,n);
    
	runtime = (clock()/(double)CLOCKS_PER_SEC ) - runtime;

	//Print the output matrix
	if (isPrint==1)
	{
        
		cout<< "L matrix:" << endl;
		PrintMatrix(l,n);
        
        // can just print matrix X after Gaussian elimination
        // This is equivalent to the upper matrix in LU decomp
        cout<< "U matrix:" << endl;
        PrintMatrix(a,n);
	}
	cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << runtime << " seconds\n";
	
	DeleteMatrix(a,n);	
	DeleteMatrix(l,n);

	return 0;
}
