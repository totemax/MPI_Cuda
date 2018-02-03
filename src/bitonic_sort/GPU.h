#ifndef __GPU_H__
#define __GPU_H__

#define THREADS_PER_BLOCK 1024

void bitonic_sort();

int *init_cuda(int size);

void end_cuda();

#endif
