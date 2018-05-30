#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
extern "C" {
    #include "GPU.h"
    #include "plano2D.h"
}

#define THREADS_PER_BLOCK 8
#define MAX_ROWS_PER_KERNEL 8


static unsigned int *colores_d;
#ifndef __CUDA_SHARED_MEM__
static unsigned int *colores;
#endif
static dim3 dimGrid, dimBlock;
static int line_width;
static int num_lines;
static int _max_iterations = 256;
static int _num_processes;
static int _my_pid;
static int lines_to_proc;
static int num_kernels;

/**
* mandelKernel
* mandelbrot pixel image processing
**/
__global__ void mandelKernel(double planoFactorXd, double planoFactorYd, double planoVxd, double planoVyd, int maxIteracionesd, unsigned int *coloresd, int img_width, int img_height, int num_processes, int my_pid, int rw) {
    int columna, fila;
    double X, Y;
    double pReal = 0.0;
    double pImag = 0.0;
    double pRealAnt, pImagAnt, distancia;

    // Determine pixel
    columna = blockIdx.x * blockDim.x + threadIdx.x;
    fila = (rw * MAX_ROWS_PER_KERNEL) + (blockIdx.y * blockDim.y) + threadIdx.y;
    int real_row = (fila * num_processes) + my_pid;

    if(real_row >= img_height)
        return;

    // Real pixel coords
    X = (planoFactorXd * (double)columna) + planoVxd;
    Y = (planoFactorYd * ((double)(img_height - 1) - (double)real_row)) + planoVyd;
    int i = 0;
    do {
        pRealAnt = pReal;
        pImagAnt = pImag;
        pReal = ((pRealAnt * pRealAnt) - (pImagAnt * pImagAnt)) + X;
        pImag = (2.0 * (pRealAnt * pImagAnt)) + Y;
        i++;
        distancia = pReal*pReal + pImag*pImag;
    }while ((i < maxIteracionesd) && (distancia <= 4.0));
    if(i == maxIteracionesd) i = 0;
    coloresd[(fila * img_width) + columna] = i;
}

extern "C" {

    /**
    * GPU_mandelInit
    * Init GPU mandelbrot
    */
    void GPU_mandelInit(int max_iterations, int img_width, int img_height, int num_processes, int my_pid){
        // Init internal vars
        line_width = img_width;
        num_lines = img_height;
        _num_processes = num_processes;
        _my_pid = my_pid;
        _max_iterations = max_iterations;
        lines_to_proc = ceil(num_lines / (float)num_processes);
        num_kernels = ceil(lines_to_proc / (float)MAX_ROWS_PER_KERNEL);

        // Init grid structure vars
        dimBlock = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        dimGrid = dim3(line_width / THREADS_PER_BLOCK, ceil(MAX_ROWS_PER_KERNEL / (float)THREADS_PER_BLOCK));

        // init GPU data structure
        #ifndef __CUDA_SHARED_MEM__
        assert(cudaMalloc((void**)&colores_d, img_width * lines_to_proc * sizeof(int)) == 0);
        colores = (unsigned int *)malloc(img_width * lines_to_proc * sizeof(int));
        #else
        assert(cudaMallocManaged((void**)&colores_d, img_width * lines_to_proc * sizeof(int)) == 0);
        #endif
    }

    /**
    * GPU_finalize
    * deinit GPU
    */
    void GPU_finalize(){
        cudaFree(colores_d);
        #ifndef __CUDA_SHARED_MEM__
        free(colores);
        #endif
        cudaDeviceReset();
    }

    /**
    * process_mandelbrot
    * GPU mandelbrot processing
    */
    unsigned int process_mandelbrot(unsigned int **colours){
        for(int i = 0; i < num_kernels; i++){
            mandelKernel<<<dimGrid, dimBlock>>>(planoFactorX, planoFactorY, planoVx, planoVy, _max_iterations, colores_d, line_width, num_lines, _num_processes, _my_pid, i);
            int result = cudaDeviceSynchronize();
            assert(result == 0);
        }
        #ifndef __CUDA_SHARED_MEM__
        assert(cudaMemcpy(colores, colores_d, lines_to_proc * line_width * sizeof(int), cudaMemcpyDeviceToHost) == 0);
        *colours = colores;
        #else
        *colours = colores_d;
        #endif
        return lines_to_proc * line_width;
    }
}
