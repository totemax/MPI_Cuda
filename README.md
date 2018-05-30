# MPI_Cuda
MPI + CUDA integration example for hybrid computing.

# Author

* Jesús Jorge Serrano - UPM (j.jorge@alumnos.upm.es)
* Plano2D and Mapapixel modules developed by Pablo Carazo

# Third parties
SHA256 module based on previous work by [Matej](https://github.com/Horkyze/CudaSHA256), also based on [CudaMiner by Christian Buchner](https://github.com/cbuchner1/CudaMiner) licensed under GPLv2.

# Introduction
This is a Computer Engineering Final Project that want to implement multiple parallel algorithms examples using the hybrid computing paradigm. In this project we will use the General Purpouse CPUs parallel computing paradigm (using MPI) and Graphics GPUs (Using CUDA), providing an cluster of GPUs connected using MPI and a local network.

## Getting Started

### Prerequisites
First of all, you need a cluster of computers (2 or more) in a local network. All of them must have a NVIDIA GPU with CUDA capability.

Following dependencies are needed in ALL the cluster computers:
* CUDA 8.0
* An MPI implementation.
* MPI c compiler.

You can use the MPI implementation that you like (OpenMPI, MPITCH, etc...).

CUDA may be installed following the steps of the [NVidia Official Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### Compiling

You can compile the project using make

#### Note

If you have the MPI compiler or NVIDIA compiler in a different path, you must override the makefile variables. For example:

```
make NVCC=/usr/local/cuda/nvcc
```

or

```
make MPICC=/usr/local/mpi/bin/mpicc
```

#### Note:

If your MPI implementation doesn't provide an c++ compiler, you must add the following  ldflag:

```
-lstdc++
```
