nvcc -std c++17 -lpthread -arch=sm_86 --default-stream per-thread -O3 ycsb.cu -o ycsb
./ycsb