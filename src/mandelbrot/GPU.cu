#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
extern "C" {
    #include "GPU.h"
    #include "plano2D.h"
}

#define THREADS_PER_BLOCK 16


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


__global__ void mandelKernel(double planoFactorXd, double planoFactorYd, double planoVxd, double planoVyd, int maxIteracionesd, unsigned int *coloresd, int img_width, int img_height, int num_processes, int my_pid) {
    int columna, fila;
    double X, Y;
    double pReal = 0.0;
    double pImag = 0.0;
    double pRealAnt, pImagAnt, distancia;
    // Determinar pixel
    columna = blockIdx.x * blockDim.x + threadIdx.x;
    fila = blockIdx.y * blockDim.y + threadIdx.y;

    int real_row = (fila * num_processes) + my_pid;

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

    void GPU_mandelInit(int max_iterations, int img_width, int img_height, int num_processes, int my_pid){
        line_width = img_width;
        num_lines = img_height;
        int lines_to_proc = num_lines / num_processes;
        _num_processes = num_processes;
        _my_pid = my_pid;
        _max_iterations = max_iterations;
        dimGrid = dim3(img_width / THREADS_PER_BLOCK, lines_to_proc / THREADS_PER_BLOCK);
        dimBlock = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        #ifndef __CUDA_SHARED_MEM__
        assert(cudaMalloc((void**)&colores_d, img_width * lines_to_proc * sizeof(int)) == 0);
        colores = (unsigned int *)malloc(img_width * lines_to_proc * sizeof(int));
        #else
        assert(cudaMallocManaged((void**)&colores_d, img_width * lines_to_proc * sizeof(int)) == 0);
        #endif
    }

    void GPU_finalize(){
        cudaFree(colores_d);
        #ifndef __CUDA_SHARED_MEM__
        free(colores);
        #endif
    }

    unsigned int process_mandelbrot(unsigned int **colours){
        int lines_to_proc = num_lines / _num_processes;
        printf("Starting machine %d, num_lines %d\n", _my_pid, lines_to_proc);
        mandelKernel<<<dimGrid, dimBlock>>>(planoFactorX, planoFactorY, planoVx, planoVy, _max_iterations, colores_d, line_width, num_lines, _num_processes, _my_pid);
        assert(cudaDeviceSynchronize() == 0);
	printf("Acabado\n");
        #ifndef __CUDA_SHARED_MEM__
        int val = cudaMemcpy(colores, colores_d, lines_to_proc * line_width * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Result: %d\n", val);
        assert(val == 0);
        *colours = colores;
        #else
        //*colours = colores_d;
        #endif
        return lines_to_proc * line_width;
    }
}
