/*
* GPU.h
* Bitonic sort GPU definitions
*/

#ifndef __GPU_H__
#define __GPU_H__

#define THREADS_PER_BLOCK 1024

/*
* bitonic_sort
* bitonic sort based in GPU. Front GPU function
*/
void bitonic_sort();

/*
* init_cuda
* Init cuda variables and data structures.
*/
int *init_cuda(int size);

/*
* end_cuda
* Deinit cuda variables and data structures.
*/
void end_cuda();

#endif
