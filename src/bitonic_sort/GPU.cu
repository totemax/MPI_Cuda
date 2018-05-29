#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

extern "C" {
    #include "GPU.h"
}

/*
* bitonic_kernel
* bitonic sort comparison implemented in CUDA
*/
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

    int *vect, *cuda_vect, num_items;

    /*
    * init_cuda
    * Init cuda variables and data structures.
    */
    int *init_cuda(int size){
        num_items = size;
        #ifdef __CUDA_SHARED_MEM__
            // create vector in GPU common memory
            cudaMallocManaged(&cuda_vect, size * sizeof(int));
            return cuda_vect;
        #else
            // create vector in GPU memory and RAM memory
            cudaMalloc(&cuda_vect, sizeof(int) * size);
            vect = (int *)malloc(sizeof(int)*size);
            return vect;
        #endif
    }

    /*
    * end_cuda
    * Deinit cuda variables and data structures.
    */
    void end_cuda(){
        cudaFree(cuda_vect);
        #ifndef __CUDA_SHARED_MEM__
            free(vect);
        #endif
        // reset cuda device
        cudaDeviceReset();
    }

    /*
    * bitonic_sort
    * bitonic sort based in GPU. Front GPU function
    */
    void bitonic_sort(){
        int num_blocks = (num_items / THREADS_PER_BLOCK) / 2;
        #ifndef __CUDA_SHARED_MEM__
            // Copy numbers to GPU memory
            cudaMemcpy(cuda_vect, vect, sizeof(int) * num_items, cudaMemcpyHostToDevice);
        #endif
        // Bitonic sort steps an jumps
        for(int step = 1; step < (num_items); step *= 2){
            for(int j = step; j > 0; j /= 2){
                bitonic_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(cuda_vect, step, j);
                cudaDeviceSynchronize();
            }
        }
        #ifndef __CUDA_SHARED_MEM__
            // Copy result to main memory
            cudaMemcpy(vect, cuda_vect, sizeof(int)*num_items, cudaMemcpyDeviceToHost);
        #endif
    }
}
