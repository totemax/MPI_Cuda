#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

extern "C" {
    #include "GPU.h"
}

__global__ void bitonic_kernel(int *vect, int step, int jump){
    int itm = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iA = (itm / jump) * jump * 2 + (itm % jump);
    int iB = iA + jump;
    unsigned char asc = ((((iA / (step * 2))) % 2) == 0);
    if((asc && (vect[iA] > vect[iB]) || (!asc && (vect[iA] < vect[iB])))){
        int tmp = vect[iA];
        vect[iA] = vect[iB];
        vect[iB] = tmp;
    }
}

extern "C" {
    void bitonic_sort(int *items, int num_items){
        int num_blocks = num_items / THREADS_PER_BLOCK;
        int *cuda_vect;

        cudaMalloc((void**)&cuda_vect, sizeof(int) * num_items);
        cudaMemcpy(cuda_vect, items, sizeof(int) * num_items, cudaMemcpyHostToDevice);
        for(int step = 1; step < (num_items / 2); step *= 2){
            for(int j = step; j > 0; j /= 2){
                bitonic_kernel<<num_blocks, THREADS_PER_BLOCK>>(cuda_vect, step, j);
                assert(cudaDeviceSynchronize() == 0);
            }
        }
        cudaMemcpy(items, cuda_vect, sizeof(int)*num_items, cudaMemcpyDeviceToHost);
        cudaFree(cuda_vect);
    }
}
