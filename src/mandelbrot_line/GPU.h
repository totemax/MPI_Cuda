#ifndef __GPU_H__
#define __GPU_H__

extern unsigned int process_mandelbrot(unsigned int **colours);

extern void GPU_mandelInit(int max_iterations, int img_width, int img_height, int num_processes, int my_pid);

extern void GPU_finalize();

#endif
