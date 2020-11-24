
__global__ void LUDecomp(float **a, float **lower, float **upper, int size){
	//Placeholder junk
	int Row = blockIdx.x*TILE + threadIdx.x;
	int Col = blockIdx.y*TILE + threadIdx.y;
}