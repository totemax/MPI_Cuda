#include "mpi.h"
#include "GPU.h"
#include "mapapixel.h"
#include "plano2D.h"

#include <sys/time.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMAGE_WIDTH 960
#define IMAGE_HEIGHT 960

#define COLOR_DEPTH (6)

#define MAX_SLAVES  (IMAGE_WIDTH)

unsigned int num_slaves, num_proc;

static tipoMapa miMapa;

typedef enum {zoomIn, zoomOut, noZoom} t_zoom;

static double _height =  0.000305;
static double _x_center  = -0.813997;
static double _y_center  =  0.194129;

void slave(void){
    short row, column, params[3];
    unsigned int *colours;
    struct timeval t0, t1, t;
    t_zoom zoom;
    MPI_Status status;

    mapiProfundidadColor (COLOR_DEPTH);

    GPU_mandelInit (mapiNumColoresDefinidos(), IMAGE_WIDTH, IMAGE_HEIGHT, num_slaves, num_proc - 1);
    planoMapear(IMAGE_WIDTH, IMAGE_HEIGHT, _x_center, _y_center, _height);

    MPI_Bcast(&params, 3, MPI_SHORT, 0, MPI_COMM_WORLD);
    while(params[0] != -1){
        row = params[0];
        column = params[1];
        zoom = params[2];
        planoPixelAPunto(row, column, &_x_center, &_y_center);
        switch(zoom){
            case zoomIn: _height = _height / 2.0; break;
            case zoomOut: _height = _height * 2.0; break;
            default: break;
        }
        planoMapear(IMAGE_WIDTH, IMAGE_HEIGHT, _x_center, _y_center, _height);
        assert (gettimeofday(&t0, NULL) == 0);
        int num_items = process_mandelbrot(&colours);
        assert (gettimeofday(&t1, NULL) == 0);
        timersub(&t1, &t0, &t);
        printf("Finished proccess of machine %d in %ld:%ld (seg:mseg)\n", num_proc, t.tv_sec, t.tv_usec/1000);
        MPI_Send(colours, num_items, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Bcast(&params, 3, MPI_SHORT, 0, MPI_COMM_WORLD);
    }
    GPU_finalize();
    MPI_Finalize();
    exit(0);
}

void end_evt(void){
    short params[3];
    params[0] = -1;
    MPI_Bcast(params, 3, MPI_SHORT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    exit(0);
}

void draw(short row, short column, short zoom){
    short actual_row = 0;
    short params[3];
    int _row, _col;
    MPI_Status status;
    tipoRGB colorRGB;

    params[0] = row;
    params[1] = column;
    params[2] = zoom;

    MPI_Bcast(params, 3, MPI_SHORT, 0, MPI_COMM_WORLD);

    int num_rows = ceil(IMAGE_HEIGHT / (float)num_slaves);
    int row_colours[num_rows * IMAGE_WIDTH];
    int slaves_sended = 0;
    struct timeval t0, t1, t;
    assert (gettimeofday(&t0, NULL) == 0);
    while(slaves_sended < num_slaves){
        MPI_Recv(row_colours, num_rows * IMAGE_WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        slaves_sended++;
        int source = status.MPI_SOURCE;
        for(_row = 0; _row < num_rows; _row++){
            int real_row = (_row * num_slaves) + (source - 1);
            if(real_row >= IMAGE_HEIGHT)
                continue;
            for(_col = 0; _col < IMAGE_WIDTH; _col++){
                mapiColorRGB (row_colours[(_row * IMAGE_WIDTH) + _col], &colorRGB);
                mapiPonerPuntoRGB (&miMapa, real_row, _col, colorRGB);
            }
        }
    }
    assert (gettimeofday(&t1, NULL) == 0);
    timersub(&t1, &t0, &t);
    mapiDibujarMapa(&miMapa);
    printf("Tiempo => %ld:%ld (seg:mseg)\n", t.tv_sec, t.tv_usec/1000);
}

void click_evt(short row, short column, int left_btn){
    t_zoom zoom;
    if(left_btn) zoom = zoomIn;
    else zoom = zoomOut;
    draw(row, column, zoom);
}

void btn_evt() {
    t_zoom zoom = noZoom;
    draw(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, zoom);
}

int main(int argc, char **argv){
    setbuf (stdout, NULL);

    unsigned char hostname[200];
    unsigned int len;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_slaves);
    MPI_Get_processor_name(hostname, &len);
    num_slaves--;

    miMapa.elColor  = colorRGB;
    miMapa.filas    = IMAGE_HEIGHT;
    miMapa.columnas = IMAGE_WIDTH;
    mapiCrearMapa (&miMapa);

    MPI_Comm_rank(MPI_COMM_WORLD, &num_proc);

    if (num_slaves > MAX_SLAVES){
        if(!num_proc) printf("Error: Sobrepasado el numero de procesos\n");
        return 0;
    }

    printf("Start process %d@%s\n", num_proc, hostname);

    if(!num_proc){
        mapiProfundidadColor(COLOR_DEPTH);
        mapiInicializar(IMAGE_WIDTH, IMAGE_HEIGHT, btn_evt, click_evt,end_evt);
    }else{
        slave();
    }
}
