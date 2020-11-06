//-----------------------------------------------------------------------
// LU Decomp - OpenMP
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
//   Get user input for matrix dimension or printing option
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
        {    cout << "Number of threads must be larger than 0" <<endl;
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

//-----------------------------------------------------------------------
//Initialize the value of matrix a[n x n]
//-----------------------------------------------------------------------
void InitializeMatrix(float** &a,int n)
{
    a = new float*[n];
    a[0] = new float[n*n];
    for (int i = 1; i < n; i++)    a[i] = a[i-1] + n;

    #pragma omp parallel for schedule(static)
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            if (i == j)
              a[i][j] = (((float)i+1)*((float)i+1))/(float)2;
            else
              a[i][j] = (((float)i+1)+((float)j+1))/(float)2;
        }
    }
}
//-----------------------------------------------------------------------
//Initialize the value of matrix to 0's
//-----------------------------------------------------------------------
void InitializeMatrixZeros(float** &a,int n)
{
    a = new float*[n];
    a[0] = new float[n*n];
    for (int i = 1; i < n; i++)    a[i] = a[i-1] + n;

    #pragma omp parallel for schedule(static)
    for (int i = 0 ; i < n ; i++)
    {
        for (int j = 0 ; j < n ; j++)
        {
            a[i][j] = 0;
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
//Print matrix
//------------------------------------------------------------------
void PrintMatrix(float **a, int n)
{
    for (int i = 0 ; i < n ; i++)
    {
        cout<< "Row " << (i+1) << ":\t" ;
        for (int j = 0 ; j < n ; j++)
        {
            printf("%.2f\t", a[i][j]);
        }
        cout<<endl ;
    }
}
//------------------------------------------------------------------
//Do LU Decomp
//------------------------------------------------------------------
void LUdecomposition(float** a, float** &l, int n)
{
    int i, j, k;
    float pivot, temp;
    
    #pragma omp parallel shared(a, l) firstprivate(n) private(pivot, i, j, temp)
    {
        for (i = 0; i < n; i++)
        {
            pivot = -1.0/a[i][i];
            
            #pragma omp single
            {
                l[i][i] = 1;
            }
            
            #pragma omp for schedule(static)
            for (k = i+1; k < n; k++)
            {
                temp = pivot*a[k][i];
                l[k][i] = a[k][i]/a[i][i];
                for (j = i; j < n; j++)
                {
                    a[k][j] = a[k][j] + temp * a[i][j];
                }
            }
            
        }
    }
}
//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int n,numThreads,isPrintMatrix;
    float **a;
    float **l;
    double runtime;
    bool isOK;
    
    if (GetUserInput(argc,argv,n,numThreads,isPrintMatrix)==false) return 1;
    
    omp_set_num_threads(numThreads);

    cout << "Starting sequential LU decomposition" << endl;
    cout << "matrix size = " << n << "x " << n << endl;
    
    //Initialize the value of matrix a, l
    InitializeMatrix(a, n);
    InitializeMatrixZeros(l, n);

    //Print the input matrices
    if (isPrintMatrix)
    {
        cout<< "Matrix A:" << endl;
        PrintMatrix(a,n);
    }

    runtime = omp_get_wtime();

    LUdecomposition(a,l,n);
    
    runtime = omp_get_wtime() - runtime;

    //Print the output matrix
    if (isPrintMatrix)
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
