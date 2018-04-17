#ifndef __GPU_H__
#define __GPU_H__

    #define NUM_CHARS 64
    #define SHA256_DIGEST_SIZE_CUDA 32

    typedef unsigned char BYTE;             // 8-bit byte

    void pre_sha256();
    int find_sha256(int num_chars, unsigned char *search, char *result);
    void end_cuda();

#endif
