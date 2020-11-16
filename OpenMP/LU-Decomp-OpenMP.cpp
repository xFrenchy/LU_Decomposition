//-----------------------------------------------------------------------
// Parallel LU Decomposition - C++ OpenMP
//-----------------------------------------------------------------------
// Programming by Tobby Lie, Anthony Dupont, Trystan Kaes, Marcus Gallegos
// Update in 11/16/2020
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

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
float* InitializeVector(int n, float value)
{

    // allocate square 2d matrix
    float *x = new float[n];
    
    #pragma omp parallel for schedule(static)
    for (int j = 0 ; j < n ; j++)
    {
        if (value == 1)  // generate input matrices (a and b)
            x[j] = (float)((rand()%10 + 1)/(float)2);
        else
            x[j] = 0;  // initializing resulting matrix
    }
    

    return x ;
}

//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
float* DeleteVector(float* x, int n)
{
    delete[] x;
}
//-----------------------------------------------------------------------
//Initialize the value of matrix x[n x n]
//-----------------------------------------------------------------------
float* PrintVector(float* x, int n)
{
    for (int j = 0 ; j < n ; j++)
    {
        cout << setiosflags(ios::fixed) << setprecision(2) << x[j] << " ";
    }
    cout << endl ;
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
//Do Forward Substitution Decomp
//------------------------------------------------------------------
float* forward_substitution(float** l, float* y, float* b, int n)
{
    int i, j;
    
    // initialize y vector to b vector
    #pragma omp parallel shared(b, y) private(i)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++)
        {
            y[i] = b[i];
        }
    }
    
    // because in the lower triangle matrix, the diagonal is all 1's
    // we can just take each element in vector y and subtract
    // it's corresponding row elements in matrix l except for the
    // diagonal element
    for (i = 1; i < n; i++)
    {
        #pragma omp parallel shared(l, y) private(j)
        {
            #pragma omp for schedule(static)
            for (j = i; j < n; j++)
            {
                y[j] = y[j] - l[j][i-1]*y[i-1];
            }
        }
    }
    
    return y;
}

//------------------------------------------------------------------
//Do Backward Substitution Decomp
//------------------------------------------------------------------
float* backward_substitution(float** u, float* x, float* y, int n)
{
    int i, j, k;
    
    x[n - 1] = y[n - 1] / u[n - 1][n - 1]; // get very last element
    
    float **temp_vec = new float*[n];
    
    for (i = 0; i < n; i++)
        temp_vec[i] = new float[n];
    
    // copy upper into temp_vec so that the contents of upper can be
    // preserved to print later
    #pragma omp parallel shared(temp_vec, u) private(i, j)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                temp_vec[i][j] = u[i][j];
            }
        }
    }
    
    // traverse columns
    for(i = n-1; i > 0; i--)
    {
        // traverse rows
        #pragma omp parallel shared(temp_vec, x) private(j, k)
        {
            // populate every other element in same column with product
            #pragma omp for schedule(static)
            for(j = i - 1; j >= 0; j--)
            {
                temp_vec[j][i] = temp_vec[j][i] * x[i];
            }
        
            // assign current element in x vector to corresponding
            // element in y vector
            #pragma omp single
            {
                x[i - 1] = y[i - 1];
            }
            
            // Solve for that row's x value
            #pragma omp for schedule(static)
            for (int k = i; k < n; k++)
            {
                x[i - 1] -= temp_vec[i-1][k];
            }
        }
        
        // finally divide x value by the corresponding element
        // in temp_vec
        x[i - 1] = x[i - 1] / temp_vec[i - 1][i - 1];
    }
    
    return x;
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

    cout << "Starting OpenMP LU decomposition" << endl;
    cout << "matrix size = " << n << "x " << n << endl;
    
    //Initialize the value of matrix a, l
    InitializeMatrix(a, n);
    InitializeMatrixZeros(l, n);
    
    float *x = InitializeVector(n, 0.0);
    float *y = InitializeVector(n, 0.0);
    float *b = InitializeVector(n, 1.0);

    //Print the input matrices
    if (isPrintMatrix)
    {
        cout<< "Matrix A:" << endl;
        PrintMatrix(a,n);
        
        cout<< "Vector b:" << endl;
        PrintVector(b,n);
    }

    runtime = omp_get_wtime();

    LUdecomposition(a,l,n);
    
    y = forward_substitution(l, y, b, n);
    
    x = backward_substitution(a, x, y, n); // here we use a as the upper
    
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
        
        cout<< "y vector:" << endl;
        PrintVector(y, n);
        
        cout<< "x vector:" << endl;
        PrintVector(x, n);
    }
    cout<< "Program runs in " << setiosflags(ios::fixed) << setprecision(8) << runtime << " seconds\n";
    
    DeleteMatrix(a,n);
    DeleteMatrix(l,n);

    return 0;
}
