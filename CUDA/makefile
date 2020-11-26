all:
	nvcc -use_fast_math -Xptxas -O3 -arch=sm_30 -o ${PWD}/LU-Decomp-CUDA ${PWD}/LU-Decomp-CUDA.cu

clean: 
	rm ${PWD}/LU-Decomp-CUDA