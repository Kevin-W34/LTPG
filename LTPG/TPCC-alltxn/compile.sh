nvcc -std c++17 -lpthread -arch=sm_86 --default-stream per-thread -O3 tp.cu -o tp
./tp