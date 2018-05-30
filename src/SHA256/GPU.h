/**
* GPU.h
*
* SHA256 CUDA implementation
**/

#ifndef __GPU_H__
#define __GPU_H__

    #define NUM_CHARS 64
    #define SHA256_DIGEST_SIZE_CUDA 32

    typedef unsigned char BYTE;             // 8-bit byte

    /**
    * pre_sha256
    * init SHA data structures.
    **/
    void pre_sha256();

    /**
    * find_sha256
    * CUDA SHA256 decrypt algorithm.
    **/
    int find_sha256(int num_chars, unsigned char *search, char *result);

    /**
    * end_cuda
    * deinit cuda
    **/
    void end_cuda();

#endif
