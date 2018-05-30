/**
* GPU.h
* Mandelbrot GPU Implementation headers
**/
#ifndef __GPU_H__
#define __GPU_H__

/**
* GPU_mandelInit
* Init GPU mandelbrot
*/
extern void GPU_mandelInit(int max_iterations, int img_width, int img_height, int num_processes, int my_pid);

/**
* process_mandelbrot
* GPU mandelbrot processing
*/
extern unsigned int process_mandelbrot(unsigned int **colours);

/**
* GPU_finalize
* deinit GPU
*/
extern void GPU_finalize();

#endif
