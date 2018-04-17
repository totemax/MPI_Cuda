#include "GPU.h"
#include "mpi.h"
#include <stdio.h>

#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "openssl/sha.h"
#include <math.h>

#define CHARS_WORD 4

unsigned int num_slaves, num_proc;

static void master(char *file, int num_words);
static void slave(int cpu_process);

int main(int argc, char **argv){

    setbuf (stdout, NULL);
    unsigned char hostname[200];
    unsigned int len;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_slaves);
    MPI_Get_processor_name(hostname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_proc);

    num_slaves--;

    if(argc < 4){
        printf("Usage: %s num_words words_file cpu_process\n", argv[0]);
        exit(0);
    }

    if(num_proc == 0){
        master(argv[2], atoi(argv[1]));
    }else{
        slave(atoi(argv[3]));
    }
    MPI_Finalize();
    exit(0);
}

static void slave(int cpu_process){
    unsigned char inv_data[SHA256_DIGEST_LENGTH], recv_data[SHA256_DIGEST_LENGTH], res_digest[SHA256_DIGEST_LENGTH];
    char result[CHARS_WORD];
    MPI_Status status;
    SHA256_CTX ctx;
    int i, c;
    memset(inv_data, 0xFF, SHA256_DIGEST_LENGTH);
    if(!cpu_process){
        pre_sha256();
    }
    while(1){
        MPI_Recv(recv_data, SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(!memcmp(recv_data, inv_data, SHA256_DIGEST_LENGTH)){
            break;
        }
        if(cpu_process){
            int num_words = pow(64, CHARS_WORD);
            char s_word[CHARS_WORD + 1] = "";
            for(i = 0; i < num_words; i++){
                int divider = 1;
                for(c = 0; c < CHARS_WORD; c++){
                    s_word[c] = 0x40 + ((i / divider) % 64);
                    divider = divider * 64;
                }
                SHA256_Init(&ctx);
                SHA256_Update(&ctx, s_word, CHARS_WORD);
                SHA256_Final(res_digest, &ctx);
                if(!memcmp(res_digest, recv_data, SHA256_DIGEST_LENGTH)){
                    memcpy(result, s_word, CHARS_WORD);
                    break;
                }
            }
        }else{
            find_sha256(CHARS_WORD, recv_data, result);
        }
        //printf("Result: %s\n", result);
        MPI_Send(result, CHARS_WORD, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    if(!cpu_process){
        end_cuda();
    }
}

static void master(char *file, int num_words){
    FILE *fd = fopen(file, "r");
    int readed_words = 0, i;

    unsigned char inv_data[SHA256_DIGEST_LENGTH];
    memset(inv_data, 0xFF, SHA256_DIGEST_LENGTH);

    if(fd == NULL){
        for(i = 0; i < num_slaves; i++){
            // Send end
            MPI_Send(&inv_data, SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, i + 1, 0, MPI_COMM_WORLD);
        }
    }


    char word[CHARS_WORD + 1] = "";
    SHA256_CTX ctx;
    unsigned char sha_datas[num_words][SHA256_DIGEST_LENGTH];
    for(readed_words = 0; readed_words < num_words; readed_words++){
        if(fscanf(fd, "%s\n", word) < 1){
            break;
        }
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, word, CHARS_WORD);
        SHA256_Final(sha_datas[readed_words], &ctx);
    }

    printf("Sending %d SHA's\n", readed_words);
    char recv_val[CHARS_WORD + 1] = "";
    MPI_Status status;

    struct timeval t0, tf, t;
    assert (gettimeofday (&t0, NULL) == 0);
    for(i = 0; i < num_slaves; i++){
        MPI_Send(sha_datas[--readed_words], SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, i + 1, 0, MPI_COMM_WORLD);
    }

    while(readed_words > 0){
        MPI_Recv(recv_val, CHARS_WORD, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Send(sha_datas[--readed_words], SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
    }

    for(i = 0; i < num_slaves; i++){
        MPI_Recv(recv_val, CHARS_WORD, MPI_CHAR, i + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Send(inv_data, SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, i + 1, 0, MPI_COMM_WORLD);
    }
    assert (gettimeofday (&tf, NULL) == 0);
    timersub (&tf, &t0, &t);
    printf ("Tiempo total GPU (seg:mseg): %ld:%ld\n", t.tv_sec, t.tv_usec/1000);
}
