/**
    Research 4 Fun

    metaCuda.cu

    Purpose: Calculates the n-th Fibonacci number an the Factorial of a number
    from CUDA + Template Meta-Programming

    @author O. A. Riveros
    @version 1.0 28 May 2014 Santiago Chile.
*/

#include <iostream>
#include <ctime>
using namespace std;

// Begin CUDA

///////////////
// Fibonacci //
///////////////
template<unsigned long N>
__device__ unsigned long  cuMetaFibonacci() {
	return  cuMetaFibonacci<N - 1>() + cuMetaFibonacci<N - 2>();
}

template<>
__device__ unsigned long  cuMetaFibonacci<0>() {
	return 1;
}

template<>
__device__ unsigned long  cuMetaFibonacci<1>() {
	return 1;
}

template<>
__device__ unsigned long  cuMetaFibonacci<2>() {
	return 1;
}

template<unsigned long N>
__global__ void cuFibonacci(unsigned long *out) {
	*out = cuMetaFibonacci<N>();
}

///////////////
// Factorial //
///////////////
template<unsigned long N>
__device__ unsigned long  cuMetaFactorial() {
	return  N * cuMetaFactorial<N - 1>();
}

template<>
__device__ unsigned long  cuMetaFactorial<1>() {
	return 1;
}

template<unsigned long N>
__global__ void cuFactorial(unsigned long *out) {
	*out = cuMetaFactorial<N>();
}

// End CUDA

int main() {

	///////////////
	// Fibonacci //
	///////////////

	size_t size = sizeof(unsigned long);

	unsigned long h_out[] = { 0 };
	unsigned long *d_out;

	cudaMalloc((void **) &d_out, size);
	cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);

	clock_t startTime = clock();

	cuFibonacci<20> <<<1, 1>>>(d_out);

	clock_t endTime = clock();
	clock_t clockTicksTaken = endTime - startTime;

	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

	cout << h_out[0] << endl;

	cudaFree(d_out);

	double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
	cout << timeInSeconds << endl;

	///////////////
	// Factorial //
	///////////////

	cudaMalloc((void **) &d_out, size);
	cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);

	startTime = clock();

	cuFactorial<20> <<<1, 1>>>(d_out);

	endTime = clock();
	clockTicksTaken = endTime - startTime;

	cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

	cout << h_out[0] << endl;

	cudaFree(d_out);

	timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
	cout << timeInSeconds << endl;

}

// Original Output
// 11:56:05 Build Finished (took 16s.185ms)
// 6765
// 4.2e-05
// 2432902008176640000
// 9e-06
