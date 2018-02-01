CC		= /usr/local/mpich-1.2.7/bin/mpicc
LD		= /usr/local/mpich-1.2.7/bin/mpicc
NVCC	= /usr/local/cuda/bin/nvcc


CCFLAGS 		= -c `pkg-config --cflags gtk+-2.0`
NVCCFLAGS 	= -c -Wno-deprecated-gpu-targets -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 --ptxas-options=-v -D__CUDA_SHARED_MEM__
LDFLAGS			= -L/usr/local/cuda/lib -lcuda -lcudart -Wl,-rpath,/usr/local/cuda/lib -lstdc++ -lm `pkg-config --libs gtk+-2.0`

SRC_DIR			= src/
OBJ_DIR			= obj/
BIN_DIR			= bin/

C_SRC 	= $(wildcard $(SRC_DIR)*/*.c)
CU_SRC 	= $(wildcard $(SRC_DIR)*/*.cu)
C_OBJ 	= $(patsubst $(SRC_DIR)%, $(OBJ_DIR)%, $(C_SRC:.c=.o))
CU_OBJ 	= $(patsubst $(SRC_DIR)%, $(OBJ_DIR)%, $(CU_SRC:.cu=.o))
OBJ			= $(C_OBJ) $(CU_OBJ)


all: mandelbrotMpiCuda mandelbrotMpiLine

mandelbrotMpiCuda: $(OBJ)
	mkdir -p $(BIN_DIR)
	$(LD) -o $(BIN_DIR)$@ $(OBJ_DIR)mandelbrot/*.o $(LDFLAGS)

mandelbrotMpiLine: $(OBJ)
	mkdir -p $(BIN_DIR)
	$(LD) -o $(BIN_DIR)$@ $(OBJ_DIR)mandelbrot_line/*.o $(LDFLAGS)

bitonic_sort: $(OBJ)
	mkdir -p $(BIN_DIR)
	$(LD) -o $(BIN_DIR)$@ $(OBJ_DIR)bitonic_sort/*.o $(LDFLAGS)

$(CU_OBJ): $(OBJ_DIR)%.o: $(SRC_DIR)%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(C_OBJ): $(OBJ_DIR)%.o: $(SRC_DIR)%.c
	mkdir -p $(dir $@)
	$(CC) -o $@ $(CCFLAGS) $<

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(OBJ_DIR)
