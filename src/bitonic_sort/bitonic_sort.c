#include "mpi.h"
#include "GPU.h"

#include <time.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static void print_result(int *vect, int num_itms);
static void master();
static void slave();

unsigned int num_slaves, num_proc;
int num_itms;

int main(int argc, char **argv){
    int max_int = 0;

    unsigned char hostname[200];
    unsigned int len;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_slaves);
    MPI_Get_processor_name(hostname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_proc);

    num_slaves--;

    if(argc < 2){
        printf("Usage: %s <num_items>\n", argv[0]);
    }

    num_itms = atoi(argv[1]);
    max_int = num_itms * 2;

    printf("num proc: %d\n", num_proc);
    if(num_proc == 0){
        master();
    }else{
        slave();
    }
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

static void master(){
    int *vect = malloc(num_itms * sizeof(int));
    int *results[4];

    MPI_Status status;

    srand(time(NULL));
    for(int i = 0; i < num_itms; i++){
        vect[i] = rand() % (num_itms * 2);
    }

    print_result(vect, num_itms);

    int items_per_slave = num_itms / num_slaves;
    for(int s = 0; s < num_slaves; s++){
        results[s] = (int *)malloc(items_per_slave * sizeof(int));
        MPI_Send(vect + (s * items_per_slave), items_per_slave, MPI_INT, s + 1, 0, MPI_COMM_WORLD);
    }

    int received_datas = 0;
    while(received_datas < num_slaves){
        MPI_Recv(results[received_datas++], items_per_slave, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    int v_idx[num_slaves];
    memset(v_idx, 0x0, sizeof(v_idx));
    for(int i = 0; i < num_itms; i++){
        int min_slave, slave_value = INT_MAX;
        for(int j = 0; j < num_slaves; j++){
            int idx = v_idx[j];
            if(idx < items_per_slave){
                if(results[j][idx] < slave_value){
                    slave_value = results[j][idx];
                    min_slave = j;
                }
            }
        }
        vect[i] = results[min_slave][v_idx[min_slave]];
        v_idx[min_slave]++;
    }

    print_result(vect, num_itms);
    free(vect);
    for(int i = 0; i < num_slaves; i++){
        free(results[i]);
    }
}

static void slave(){
    int items_per_slave = num_itms / num_slaves;
    int *vect = (int *)malloc(sizeof(int) * items_per_slave);
    MPI_Status status;
    MPI_Recv(vect, items_per_slave, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    bitonic_sort(vect, items_per_slave);
    print_result(vect, items_per_slave);
    MPI_Send(vect, items_per_slave, MPI_INT, 0, 0, MPI_COMM_WORLD);
}
