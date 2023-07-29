nvcc -g -G -std c++17 -lpthread -arch=sm_86 --default-stream per-thread ycsb.cu -o ycsb
cuda-gdb