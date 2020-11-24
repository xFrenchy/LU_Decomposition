//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
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