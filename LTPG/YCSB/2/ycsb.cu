#include <bits/stdc++.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Database.h"
#include "Query.h"
#include "Execute.h"
__global__ void KERNEL_EXECUTE(YCSB_A_SET *ycsb_a_set,
                               YCSB_A_QUERY *ycsb_a_query,
                               YCSB_LOG *log,
                               YCSB_SNAPSHOT *snapshot,
                               YCSB_INDEX *index)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    unsigned int ID = wID / (YCSB_OP_SIZE * 2);
    // execute ycsb_a_set
    // if (ID >= MINI_BATCH_CNT)
    // {
    //     return;
    // }
    YCSB_A_NAMESPACE::execute(ID * 32 + thID % 32, ycsb_a_set, ycsb_a_query, log, snapshot, index);
}
__global__ void KERNEL_CHECK(YCSB_A_SET *ycsb_a_set,
                             YCSB_A_QUERY *ycsb_a_query,
                             YCSB_LOG *log,
                             YCSB_SNAPSHOT *snapshot,
                             YCSB_INDEX *index)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    unsigned int ID = wID / (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE);
    // check ycsb_a_set
    // if (ID >= MINI_BATCH_CNT)
    // {
    //     return;
    // }
    YCSB_A_NAMESPACE::check(ID * 32 + thID % 32, ycsb_a_set, ycsb_a_query, log, snapshot, index);
}
__global__ void KERNEL_WRITEBACK(YCSB_A_SET *ycsb_a_set,
                                 YCSB_A_QUERY *ycsb_a_query,
                                 YCSB_LOG *log,
                                 YCSB_SNAPSHOT *snapshot,
                                 YCSB_INDEX *index)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    unsigned int ID = wID / (1 + YCSB_WRITE_SIZE);
    // writeback
    // if (ID >= MINI_BATCH_CNT)
    // {
    //     return;
    // }
    YCSB_A_NAMESPACE::write_back(ID * 32 + thID % 32, ycsb_a_set, ycsb_a_query, log, snapshot, index);
    if (DATA_DISTRIBUTION == ZIPFIAN)
        YCSB_A_NAMESPACE::reduce_update(ID * 32 + thID % 32, ycsb_a_set, ycsb_a_query, log, snapshot, index);
}

int main(int argc, char const *argv[])
{
    cudaSetDevice(SET_DEVICE);
    // TPCCDatabase *tpcc_database = new TPCCDatabase();
    // initial_tpcc_data(tpcc_database);
    YCSBDatabase *ycsb_database = new YCSBDatabase();
    initial_ycsb_data(ycsb_database);
    // TPCCQuery *tpcc_query = new TPCCQuery();
    // initial_tpcc_new_query(tpcc_query);
    YCSBQuery *ycsb_query = new YCSBQuery();
    initial_ycsb_new_query(ycsb_query);
    std::thread transfer_query;
    cudaStream_t stream[STREAM_SIZE];
    for (size_t i = 0; i < STREAM_SIZE; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    for (int epoch_ID = 0; epoch_ID < EPOCH_TP; epoch_ID++)
    {
        // make_tpcc_new_query(epoch_ID, tpcc_query);
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        // tpcc_query->random_choose_query(epoch_ID, stream);
        cudaError_t err;
        long long start_kernel_0_t = current_time();
        cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        ycsb_query->random_choose_query(epoch_ID, stream);

        KERNEL_EXECUTE<<<EXECUTE_GRID_SIZE, EXECUTE_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(ycsb_query->ycsb_a_set_d + epoch_ID % STREAM_SIZE,
                                                                                                     ycsb_query->ycsb_a_query_d + epoch_ID % STREAM_SIZE,
                                                                                                     ycsb_database->log,
                                                                                                     ycsb_database->snapshot_d,
                                                                                                     ycsb_database->index_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        long long end_0 = current_time();
        KERNEL_CHECK<<<CHECK_GRID_SIZE, CHECK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(ycsb_query->ycsb_a_set_d + epoch_ID % STREAM_SIZE,
                                                                                               ycsb_query->ycsb_a_query_d + epoch_ID % STREAM_SIZE,
                                                                                               ycsb_database->log,
                                                                                               ycsb_database->snapshot_d,
                                                                                               ycsb_database->index_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        long long end_1 = current_time();
        KERNEL_WRITEBACK<<<WRITEBACK_GRID_SIZE, WRITEBACK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(ycsb_query->ycsb_a_set_d + epoch_ID % STREAM_SIZE,
                                                                                                           ycsb_query->ycsb_a_query_d + epoch_ID % STREAM_SIZE,
                                                                                                           ycsb_database->log,
                                                                                                           ycsb_database->snapshot_d,
                                                                                                           ycsb_database->index_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        // if (epoch_ID >= WARMUP_TP && epoch_ID < EPOCH_TP - WARMUP_TP)
        // {
        statistic_ycsb_query(epoch_ID, ycsb_query, stream);
        // }
        long long end_t = current_time();
        float kernel_0_time = duration(start_kernel_0_t, end_t);
        float time_0 = duration(start_kernel_0_t, end_0);
        float time_1 = duration(end_0, end_1);
        float time_2 = duration(end_1, end_t);
        // std::cout << "Epoch " << epoch_ID << " kernel cost [" << kernel_0_time << " s].\n";
        // std::cout << "kernel 0 cost [" << time_0 << " s].\n";
        // std::cout << "kernel 1 cost [" << time_1 << " s].\n";
        // std::cout << "kernel 2 cost [" << time_2 << " s].\n";
        // check(tpcc_database->log);
        if (epoch_ID >= WARMUP_TP && epoch_ID < EPOCH_TP - WARMUP_TP)
        {
            ycsb_query->kernel_0_time_all += kernel_0_time;
            ycsb_query->time_0_all += time_0;
            ycsb_query->time_1_all += time_1;
            ycsb_query->time_2_all += time_2;
            // statistic_tpcc_query(epoch_ID, tpcc_query);
        }
        ycsb_database->clear_LOG();
    }
    for (size_t i = 0; i < STREAM_SIZE; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    ycsb_database->print();
    ycsb_query->print();
    float average_commit = ((float)ycsb_query->commit_all) / (EPOCH_TP - 2 * WARMUP_TP);
    float average_kernel_time = ycsb_query->kernel_0_time_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_0 = ycsb_query->time_0_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_1 = ycsb_query->time_1_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_2 = ycsb_query->time_2_all / (EPOCH_TP - 2 * WARMUP_TP);
    std::cout << "Kernel all cost [" << average_kernel_time << " s] in average.\n\n";
    std::cout << "Kernel 0 cost [" << average_time_0 << " s] in average.\n";
    std::cout << "Kernel 1 cost [" << average_time_1 << " s] in average.\n";
    std::cout << "Kernel 2 cost [" << average_time_2 << " s] in average.\n\n";
    std::cout << "memory copy speed is [" << ycsb_query->memory_speed / (EPOCH_TP - 2 * WARMUP_TP) << " GB/s].\n\n";
    std::cout << "Commit [" << average_commit << "] in average.\n\n";
    std::cout << "TPS [" << average_commit / average_kernel_time << "].\n\n";
    std::cout << "The table size is [" << YCSB_SIZE << "].";
    std::cout << "The batch size is [" << BATCH_SIZE << "].\n\n";
    free(ycsb_database);
    free(ycsb_query);
    return 0;
}
