#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "Predefine.h"
#include "Datastructure.h"
#include "Genericfunction.h"
#include "Random.h"

#define Threshold 51
class MakeYCSB
{
private:
    Random random;

public:
    MakeYCSB()
    {
    }
    ~MakeYCSB()
    {
    }
    ycsbAQuery makeYCSBAQuery_zipf()
    {
        ycsbAQuery query;
        for (size_t i = 0; i < YCSB_READ_SIZE;)
        {
            unsigned int zipf = random.uniform_dist(1, 100);
            if (zipf < Threshold)
            {
                query.LOC_R[i] = random.uniform_dist(1, 1);
            }
            else
            {
                query.LOC_R[i] = random.uniform_dist(2, YCSB_SIZE);
            }
            for (size_t j = 0; j < i; j++)
            {
                if (query.LOC_R[j] == query.LOC_R[i])
                {
                    continue;
                }
            }
            i++;
        }
        for (size_t i = 0; i < YCSB_WRITE_SIZE;)
        {
            unsigned int zipf = random.uniform_dist(1, 100);
            if (zipf < Threshold)
            {
                query.LOC_W[i] = random.uniform_dist(1, 1);
            }
            else
            {
                query.LOC_W[i] = random.uniform_dist(2, YCSB_SIZE);
            }
            for (size_t j = 0; j < i; j++)
            {
                if (query.LOC_W[j] == query.LOC_W[i])
                {
                    continue;
                }
            }
            for (size_t j = 0; j < YCSB_READ_SIZE; j++)
            {
                if (query.LOC_R[j] == query.LOC_W[i])
                {
                    continue;
                }
            }
            i++;
        }
        return query;
    }
    ycsbAQuery makeYCSBAQUERY_uniform()
    {
        ycsbAQuery query;
        for (size_t i = 0; i < YCSB_READ_SIZE;)
        {
            query.LOC_R[i] = random.uniform_dist(1, YCSB_SIZE - 1);
            for (size_t j = 0; j < i; j++)
            {
                if (query.LOC_R[j] == query.LOC_R[i])
                {
                    continue;
                }
            }
            i++;
        }
        for (size_t i = 0; i < YCSB_WRITE_SIZE;)
        {
            query.LOC_W[i] = random.uniform_dist(1, YCSB_SIZE - 1);
            for (size_t j = 0; j < i; j++)
            {
                if (query.LOC_W[j] == query.LOC_W[i])
                {
                    continue;
                }
            }
            for (size_t j = 0; j < YCSB_READ_SIZE; j++)
            {
                if (query.LOC_R[j] == query.LOC_W[i])
                {
                    continue;
                }
            }
            i++;
        }
        return query;
    }
};

std::atomic<unsigned int> YCSB_ID(0);
std::atomic<unsigned int> YCSB_A_ID(0);

class YCSBQuery
{
private:
    Random random;

public:
    YCSB_A_SET *ycsb_a_set_new;
    YCSB_A_SET *ycsb_a_set_d;
    YCSB_A_QUERY *ycsb_a_query;
    YCSB_A_QUERY *ycsb_a_query_d;
    MakeYCSB *makeycsb;

    unsigned int commit_ycsb_a = 0;
    unsigned int commit_all = 0;
    float kernel_0_time_all = 0.0;
    float time_0_all = 0.0;
    float time_1_all = 0.0;
    float time_2_all = 0.0;
    float memory_speed = 0.0;
    YCSBQuery()
    {
        cudaMallocHost((void **)&ycsb_a_query, sizeof(YCSB_A_QUERY) * PRE_GEN_EPOCH);
        cudaMallocHost((void **)&ycsb_a_set_new, sizeof(YCSB_A_SET) * STREAM_SIZE);
        cudaMalloc((void **)&ycsb_a_set_d, sizeof(YCSB_A_SET) * STREAM_SIZE);
        cudaMalloc((void **)&ycsb_a_query_d, sizeof(YCSB_A_QUERY) * STREAM_SIZE);
        makeycsb = new MakeYCSB();
    }
    ~YCSBQuery()
    {
        cudaFreeHost(ycsb_a_set_new);
        cudaFreeHost(ycsb_a_query);
        cudaFree(ycsb_a_query_d);
        cudaFree(ycsb_a_set_d);
    }
    void clear_QUERY()
    {
        memset(ycsb_a_query, 0, sizeof(YCSB_A_QUERY) * PRE_GEN_EPOCH);
    }
    void initial_ycsb_a_query(unsigned int epochID, unsigned int BATCH_ID)
    {
        for (unsigned int i = 0; i < YCSB_A_SIZE / YCSB_CPU_THREAD_SIZE;)
        {
            unsigned int q_id = YCSB_A_ID++;
            unsigned int tid = YCSB_ID++;
            ycsbAQuery query;
            if (DATA_DISTRIBUTION == ZIPFIAN)
            {
                query = makeycsb->makeYCSBAQuery_zipf();
            }
            else
            {
                query = makeycsb->makeYCSBAQUERY_uniform();
            }

            for (size_t j = 0; j < YCSB_READ_SIZE; j++)
            {
                ycsb_a_query[epochID].LOC_R[j * YCSB_READ_SIZE + q_id] = query.LOC_R[j];
            }
            for (size_t j = 0; j < YCSB_WRITE_SIZE; j++)
            {
                ycsb_a_query[epochID].LOC_W[j * YCSB_WRITE_SIZE + q_id] = query.LOC_W[j];
            }
            ycsb_a_query[epochID].TID[q_id] = (epochID & 0xfff) << 20 + tid;
            i++;
        }
    }
    void random_choose_query(unsigned int epochID, cudaStream_t *stream)
    {
        unsigned int choose = random.uniform_dist(0, PRE_GEN_EPOCH - 1);
        copy_to_device(choose, epochID, stream);
    }
    void copy_to_device(unsigned int choose, unsigned int epochID, cudaStream_t *stream)
    {
        long long start = current_time();
        cudaMemset(&ycsb_a_set_d[epochID % STREAM_SIZE], 0, sizeof(YCSB_A_SET));
        // cudaMemcpy(&neworder_set_d[epochID % 2], &neworder_set[choose], sizeof(NEWORDER_SET), cudaMemcpyHostToDevice);
        // cudaMemcpy(&payment_set_d[epochID % 2], &payment_set[choose], sizeof(PAYMENT_SET), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&ycsb_a_query_d[epochID % STREAM_SIZE], &ycsb_a_query[choose], sizeof(YCSB_A_QUERY), cudaMemcpyHostToDevice, stream[epochID % STREAM_SIZE]);
        cudaStreamSynchronize(stream[epochID % STREAM_SIZE]);
        long long end = current_time();
        float time = duration(start, end);
        // std::cout << "copy to device costs [" << time << "s].\n";
        float query_size = sizeof(YCSB_A_QUERY) / 1024;
        // std::cout << "query_aize is [" << query_size << " KB].\n";
        float speed = query_size / time / 1024 / 1024;
        if (epochID >= WARMUP_TP && epochID < EPOCH_TP - WARMUP_TP)
        {
            this->memory_speed += speed;
        }
        // std::cout << "query_aize speed is [" << speed << " GB/s].\n";
    }
    void copy_to_host(unsigned int epochID, cudaStream_t *stream)
    {
        long long start = current_time();
        cudaMemcpyAsync(&ycsb_a_set_new[epochID % STREAM_SIZE].COMMIT, &ycsb_a_set_d[epochID % STREAM_SIZE].COMMIT, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[epochID % STREAM_SIZE]);
        cudaStreamSynchronize(stream[epochID % STREAM_SIZE]);
        long long end = current_time();
        float time = duration(start, end);
        // std::cout << "copy to host costs [" << time << " s].\n";
    }
    void print()
    {
        std::cout << "YCSB_A_SET is [" << (float)sizeof(YCSB_A_SET) / 1024 / 1024 << " MB].\n\n";
        std::cout << "YCSB_A_QUERY is [" << (float)sizeof(YCSB_A_QUERY) / 1024 / 1024 << " MB].\n\n";
    }
};
void initial_ycsb_new_query(YCSBQuery *query)
{
    query->clear_QUERY();
    long long start_t = current_time();
    for (size_t epoch_ID = 0; epoch_ID < PRE_GEN_EPOCH; epoch_ID++)
    {
        std::vector<std::thread> threads_a;
        for (unsigned int i = 0; i < YCSB_CPU_THREAD_SIZE; i++)
        {
            threads_a.push_back(std::thread(&YCSBQuery::initial_ycsb_a_query, query, epoch_ID, i));
        }
        for (unsigned int i = 0; i < YCSB_CPU_THREAD_SIZE; i++)
        {
            threads_a[i].join();
        }
        YCSB_A_ID = 0;
        YCSB_ID = 0;
    }
    long long end_t = current_time();
    float time = duration(start_t, end_t);
    std::cout << "Initiallization of " << PRE_GEN_EPOCH << " YCSB queries costs [" << time << " s].\n";
}
void statistic_ycsb_query(unsigned int epochID, YCSBQuery *query, cudaStream_t *stream)
{
    long long start_t = current_time();
    // std::vector<std::thread> threads;
    // for (unsigned int i = 0; i < TPCC_CPU_THREAD_SIZE; i++)
    // {
    //     threads.push_back(std::thread(&Query::statistic, query, BATCH_ID));
    // }
    // for (unsigned int i = 0; i < TPCC_CPU_THREAD_SIZE; i++)
    // {
    //     threads[i].join();
    // }
    long long end_t = current_time();
    float time = duration(start_t, end_t);
    // analyse_n_ID = 0;
    // analyse_p_ID = 0;
    query->copy_to_host(epochID, stream);
    if (epochID >= WARMUP_TP && epochID < EPOCH_TP - WARMUP_TP)
    {
        query->commit_ycsb_a += query->ycsb_a_set_new[epochID % STREAM_SIZE].COMMIT;
        // std::cout << "commit neworder " << query->commit_n << std::endl;
        // std::cout << "commit payment " << query->commit_p << std::endl;
        query->commit_all = query->commit_ycsb_a;
    }
    // std::cout << "Analyse of queries costs [" << time << "s].\n";
}
