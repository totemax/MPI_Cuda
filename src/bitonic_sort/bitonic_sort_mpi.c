#include "mpi.h"
#include "GPU.h"

#include <time.h>
#include <sys/time.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAX_PACKET_SIZE 1048576

static void print_result(int *vect, int num_itms);
static void master();
static void slave();

unsigned int num_slaves, num_proc;
int num_itms;
unsigned char cpu = 0;

int main(int argc, char **argv){
    setbuf (stdout, NULL);
    unsigned char hostname[200];
    unsigned int len;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_slaves);
    MPI_Get_processor_name(hostname, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &num_proc);

    num_slaves--;

    if(argc < 2){
        printf("Usage: %s <num_items> [cpu]\n", argv[0]);
    }

    num_itms = atoi(argv[1]);
    if(argc > 2)
        cpu = (unsigned char)atoi(argv[2]);

    if(num_proc == 0){
        master();
    }else{
        slave();
    }
    MPI_Finalize();
    exit(0);
}

static void send_data(int *vect, int num_items, int slave){
	int i = 0;
    int items_remaining = num_items;
	while(items_remaining > 0){
        int items_send = (items_remaining > MAX_PACKET_SIZE) ? MAX_PACKET_SIZE : items_remaining;
        MPI_Send(vect + i, items_send , MPI_INT, slave, 0, MPI_COMM_WORLD);
        i += items_send;
        items_remaining -= items_send;
	}
}

static void recv_data(int *vect, int num_items){
	int i = 0;
    MPI_Status status;
    int items_remaining = num_items;
    int origin = -1;
	while(items_remaining > 0){
        int items_receive = (items_remaining > MAX_PACKET_SIZE) ? MAX_PACKET_SIZE : items_remaining;
    	MPI_Recv(vect + i, items_receive, MPI_INT, (origin < 0) ? MPI_ANY_SOURCE : origin, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if(origin < 0)
            origin = status.MPI_SOURCE;
        i += items_receive;
        items_remaining -= items_receive;
	}
}

static void print_result(int *vect, int num_itms){
    printf("Vector itms: \n");
    printf("[");
    int i = 0;
    for(i = 0; i < num_itms; i++){
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
    int *results[num_slaves];
    struct timeval t0, tf, t;
    int i = 0, j = 0, s;

    printf("Starting master process at pid %d\n", num_proc);
    MPI_Status status;
    srand(time(NULL));
    for(i = 0; i < num_itms; i++){
        vect[i] = rand() % (num_itms * 2);
    }
    int items_per_slave = num_itms / num_slaves;
    for(s = 0; s < num_slaves; s++){
        results[s] = (int *)malloc(items_per_slave * sizeof(int));
    }

    assert (gettimeofday (&t0, NULL) == 0);
    for(s = 0; s < num_slaves; s++){
        send_data(vect, items_per_slave, s+1);
    }

    int received_datas = 0;
    while(received_datas < num_slaves){
        recv_data(results[received_datas], items_per_slave);
        received_datas++;
    }

    int v_idx[num_slaves];
    memset(v_idx, 0x0, sizeof(v_idx));
    for(i = 0; i < num_itms; i++){
        int min_slave, slave_value = INT_MAX;
        for(j = 0; j < num_slaves; j++){
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
    assert (gettimeofday (&tf, NULL) == 0);

    timersub (&tf, &t0, &t);
    printf ("\nTiempo total (seg:mseg): %ld:%ld\n", t.tv_sec, t.tv_usec/1000);
    for(i = 0; i < num_itms - 1; i++){
        if(vect[i] > vect[i + 1]){
            printf("%d: Item: %d, item + 1: %d\n",  i, vect[i], vect[i+1]);
            printf("Error de ordenación en item %d\n", i);
            return;
        }
    }
    printf("Ordenacion OK\n");
    free(vect);
    for(i = 0; i < num_slaves; i++){
        free(results[i]);
    }
}


void comparar(int *vOrg, int indA, int indB, unsigned char ascendente) {
    int tmp;
    if ((ascendente && (vOrg[indA] > vOrg[indB]))
    || (!ascendente && (vOrg[indA] < vOrg[indB])) ) {
        tmp        = vOrg[indA];
        vOrg[indA] = vOrg[indB];
        vOrg[indB] = tmp;
    }
}



static void bitonic_sort_cpu(int *vect, int cardinalidad){
    int ultimoPaso = cardinalidad / 2;
    int maxEntero = cardinalidad * 2;
    unsigned char ascendente = 0;
    int paso, j, i, numComparaciones, iA, iB, salto;

    for (paso=1; paso<=ultimoPaso; paso*=2) {
        numComparaciones = 0;
        for (salto=paso; salto>0; salto/=2) {
            ascendente = 1;
            for (i=0; i<cardinalidad; i+=salto*2) {
                iA = i; iB = iA+salto;
                for (j=0; j<salto; j++) {
                    comparar(vect, iA, iB, ascendente);
                    iA++; iB++;
                    numComparaciones++;
                }
                if ((numComparaciones % paso) == 0)
                    ascendente = (ascendente ? 0 : 1);
            }
        }
    }
}

static void slave(){
    printf("Starting slave process at pid %d with cpuflag = %d\n", num_proc, cpu);
    int items_per_slave = num_itms / num_slaves, i;
    int *vect;
    if(!cpu)
        vect = init_cuda(items_per_slave);
    else
        vect = (int *)malloc(items_per_slave * sizeof(int));
    MPI_Status status;
	recv_data(vect, items_per_slave);
    if(!cpu)
        bitonic_sort();
    else
        bitonic_sort_cpu(vect, items_per_slave);
	send_data(vect, items_per_slave, 0);
    if(!cpu)
        end_cuda();
    else
        free(vect);
}
