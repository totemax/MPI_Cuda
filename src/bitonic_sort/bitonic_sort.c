#include "mpi.h"

#include <sys/time.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_result(int *vect, int num_itms);

unsigned int num_slaves, num_proc;

int main(int argc, char **argv){
    int *vect, num_itms, max_int = 0;

    unsigned char hostname[200];
    unsigned int len;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_slaves);
    MPI_Get_processor_name(hostname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_proc);

    if(argc < 2){
        printf("Usage: %s <num_items>\n", argv[0]);
    }

    num_itms = atoi(argv[1]);
    max_int = num_itms * 2;
    vect = malloc(num_itms * sizeof(int));
    srand(time(NULL));
    for(int i = 0; i < num_itms; i++){
        vect[i] = rand() % max_int;
    }

    print_result(vect, num_itms);

}

static void print_result(int *vect, int num_itms){
    printf("Vector itms: \n");
    printf("[");
    for(int i = 0; i < num_itms; i++){
        if(i == num_itms - 1){
            printf("%d", vect[i]);
        }else{
            printf("%d, ", vect[i]);
        }
    }
    printf("]\n");
}
