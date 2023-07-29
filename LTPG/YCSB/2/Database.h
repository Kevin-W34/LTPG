#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "Predefine.h"
#include "Datastructure.h"
#include "Genericfunction.h"
#include "Random.h"

class YCSBDatabase
{
private:
    YCSB_TABLE *ycsb_tbl;
    Random random;

public:
    YCSB_SNAPSHOT *snapshot;
    YCSB_SNAPSHOT *snapshot_d;
    YCSB_INDEX *index;
    YCSB_INDEX *index_d;
    YCSB_LOG *log;
    float time;
    YCSBDatabase()
    {
        cudaMallocManaged((void **)&snapshot, sizeof(YCSB_SNAPSHOT));
        cudaMallocManaged((void **)&index, sizeof(YCSB_INDEX));
        cudaMalloc((void **)&snapshot_d, sizeof(YCSB_SNAPSHOT));
        cudaMalloc((void **)&index_d, sizeof(YCSB_INDEX));
        cudaMalloc((void **)&log, sizeof(YCSB_LOG));
        ycsb_tbl = new YCSB_TABLE[YCSB_SIZE];
    }
    ~YCSBDatabase()
    {
        cudaFree(snapshot);
        cudaFree(snapshot_d);
        cudaFree(index);
        cudaFree(index_d);
        cudaFree(log);
        free(ycsb_tbl);
    }
    void clear_LOG()
    {
        cudaMemset(log->YCSB_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        cudaMemset(log->YCSB_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        cudaMemset(log->YTD, 0, sizeof(unsigned int) * YCSB_A_SIZE);
        cudaMemset(log->TMP_YTD, 0, sizeof(unsigned int) * YCSB_A_SIZE);
        // cudaMemset(log->Y_0_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_0_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_1_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_1_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_2_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_2_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_3_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_3_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_4_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_4_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_5_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_5_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_6_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_6_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_7_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_7_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_8_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_8_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_9_R, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
        // cudaMemset(log->Y_9_W, 0xffffffff, sizeof(unsigned int) * YCSB_SIZE);
    }
    void create_table(unsigned int thID)
    {
        for (size_t i = 0; i < YCSB_SIZE / YCSB_CPU_THREAD_SIZE; i++)
        {
            unsigned int ii = i + thID * YCSB_SIZE / YCSB_CPU_THREAD_SIZE;
            ycsb_tbl[ii].Y_0 = 0;
            ycsb_tbl[ii].Y_1 = 1;
            ycsb_tbl[ii].Y_2 = 2;
            ycsb_tbl[ii].Y_3 = 3;
            ycsb_tbl[ii].Y_4 = 4;
            ycsb_tbl[ii].Y_5 = 5;
            ycsb_tbl[ii].Y_6 = 6;
            ycsb_tbl[ii].Y_7 = 7;
            ycsb_tbl[ii].Y_8 = 8;
            ycsb_tbl[ii].Y_9 = 9;

            snapshot->data[0 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_0;
            snapshot->data[1 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_1;
            snapshot->data[2 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_2;
            snapshot->data[3 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_3;
            snapshot->data[4 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_4;
            snapshot->data[5 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_5;
            snapshot->data[6 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_6;
            snapshot->data[7 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_7;
            snapshot->data[8 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_8;
            snapshot->data[9 * YCSB_SIZE + ii] = ycsb_tbl[ii].Y_9;
        }
    }
    void print()
    {
        std::cout << "==========================================\n";
        std::cout << "\t YCSB_SIZE = " << YCSB_SIZE << std::endl;
        std::cout << "------------------------------------------\n";
        std::cout << "\t TABLE_SIZE     = " << YCSB_SIZE * 10 << std::endl;
        std::cout << "\t BATCH_SIZE     = " << BATCH_SIZE << std::endl;
        std::cout << "==========================================\n\n";
        std::cout << "SNAPSHOT is [" << (float)sizeof(YCSB_SNAPSHOT) / 1024 / 1024 / 1024 << "GB].\n\n";
        std::cout << "INDEX is [" << (float)sizeof(YCSB_INDEX) / 1024 / 1024 << "MB].\n\n";
        std::cout << "LOG is [" << (float)sizeof(YCSB_LOG) / 1024 / 1024 / 1024 << "GB].\n\n";
    }
};

void initial_ycsb_data(YCSBDatabase *database)
{
    long long start_t = current_time();
    std::vector<std::thread> threads;
    for (size_t i = 0; i < YCSB_CPU_THREAD_SIZE; i++)
    {
        threads.push_back(std::thread(&YCSBDatabase::create_table, database, i));
    }
    for (size_t i = 0; i < YCSB_CPU_THREAD_SIZE; i++)
    {
        threads[i].join();
    }
    cudaMemcpy(database->snapshot_d, database->snapshot, sizeof(YCSB_SNAPSHOT), cudaMemcpyHostToDevice);
    cudaMemcpy(database->index_d, database->index, sizeof(YCSB_INDEX), cudaMemcpyHostToDevice);
    long long end_t = current_time();
    database->time = duration(start_t, end_t);
    std::cout << "Initiallization of all tables of YCSB costs [" << database->time << "s].\n\n";
}