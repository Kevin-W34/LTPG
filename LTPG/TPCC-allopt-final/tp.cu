#include <bits/stdc++.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Database.h"
#include "Query.h"
// #include "Execute.h"
#include "Execute_neworder.h"
#include "Execute_payment.h"
__global__ void KERNEL_CID(PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
{ // get c_id if used c_name
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    for (size_t i = wID; i < PAYMENT_CNT; i += gridDim.x * blockDim.x / WARP_SIZE)
    {
        PAYMENT_NAMESPACE::get_c_id(i, payment_set, payment_query, log, snapshot, index);
    }
}
__global__ void KERNEL_EXECUTE(NEWORDER_SET *neworder_set,
                               PAYMENT_SET *payment_set,
                               NEWORDER_QUERY *neworder_query,
                               PAYMENT_QUERY *payment_query,
                               LOG *log,
                               SNAPSHOT *snapshot,
                               INDEX *index)
{ // execution phase
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    // while (true)
    // {
    unsigned int ID = wID / EXECUTE_WARP;
    if (wID % EXECUTE_WARP < 130)
    { // execute neworder_set
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        NEWORDER_NAMESPACE::execute(ID * 32 + thID % 32, neworder_set, neworder_query, log, snapshot, index);
    }
    else
    { // execute payment_set
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        PAYMENT_NAMESPACE::execute(ID * 32 + thID % 32, payment_set, payment_query, log, snapshot, index);
    }
    // }
}
__global__ void KERNEL_CHECK(NEWORDER_SET *neworder_set,
                             PAYMENT_SET *payment_set,
                             NEWORDER_QUERY *neworder_query,
                             PAYMENT_QUERY *payment_query,
                             LOG *log,
                             SNAPSHOT *snapshot,
                             INDEX *index)
{ // check conflicts phase
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    // while (true)
    // {
    unsigned int ID = wID / CHECK_WARP;
    if (wID % CHECK_WARP < 82)
    { // check neworder_set
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        NEWORDER_NAMESPACE::check(ID * 32 + thID % 32, neworder_set, neworder_query, log, snapshot, index);
    }
    else
    { // check payment_set
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        PAYMENT_NAMESPACE::check(ID * 32 + thID % 32, payment_set, payment_query, log, snapshot, index);
    }
    // }
}
__global__ void KERNEL_WRITEBACK(NEWORDER_SET *neworder_set,
                                 PAYMENT_SET *payment_set,
                                 NEWORDER_QUERY *neworder_query,
                                 PAYMENT_QUERY *payment_query,
                                 LOG *log,
                                 SNAPSHOT *snapshot,
                                 INDEX *index)
{ // writeback phase
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int wID = thID / 32;
    // while (true)
    // {
    unsigned int ID = wID / WRITEBACK_WARP;
    if (wID % WRITEBACK_WARP < 63)
    { // writeback
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        NEWORDER_NAMESPACE::write_back(ID * 32 + thID % 32, neworder_set, neworder_query, log, snapshot, index);
    }
    else
    { // writeback
        // if (ID >= MINI_BATCH_CNT)
        // {
        //     return;
        // }
        PAYMENT_NAMESPACE::write_back(ID * 32 + thID % 32, payment_set, payment_query, log, snapshot, index);
    }
    // }
    PAYMENT_NAMESPACE::reduce_update(thID, payment_set, payment_query, log, snapshot, index);
}

int main(int argc, char const *argv[])
{
    for (int cnt = 0; cnt < 1; cnt++)
    {
        cudaSetDevice(SET_DEVICE);
        Database *database = new Database();
        initial_data(database);
        // database->copy_to_device();
        Query *query = new Query();
        initial_new_query(query);
        std::thread transfer_query;
        cudaStream_t stream[STREAM_SIZE];
        for (size_t i = 0; i < STREAM_SIZE; i++)
        {
            cudaStreamCreate(&stream[i]);
        }

        for (int epoch_ID = 0; epoch_ID < EPOCH_TP; epoch_ID++)
        {
            // make_new_query(epoch_ID, query);
            // cudaStreamSynchronize();
            // query->random_choose_query(epoch_ID, stream);
            long long start_kernel_0_t = current_time();
            cudaError_t err;
            int slotID = epoch_ID % SLOT_SIZE;
            query->random_choose_query(epoch_ID, stream);
            cudaStreamSynchronize(stream[epoch_ID]);

            if (NEWORDER_PERCENT != 100)
            {
                // KERNEL_CID<<<GRID_SIZE, BLOCK_SIZE, 0>>>(query->payment_set_d + streamID,
                //                                          query->payment_query_d + streamID,
                //                                          database->log,
                //                                          database->snapshot_d,
                //                                          database->index_d);
                KERNEL_CID<<<GRID_SIZE, BLOCK_SIZE, 0, stream[epoch_ID]>>>(query->payment_set_d + slotID,
                                                                           query->payment_query_d + slotID,
                                                                           database->log,
                                                                           database->snapshot_d,
                                                                           database->index_d);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
                    break;
                }
                // cudaDeviceSynchronize();
                cudaStreamSynchronize(stream[epoch_ID]);
            }
            // KERNEL_EXECUTE<<<EXECUTE_GRID_SIZE, EXECUTE_BLOCK_SIZE, 0>>>(query->neworder_set_d + streamID,
            //                                                              query->payment_set_d + streamID,
            //                                                              query->neworder_query_d + streamID,
            //                                                              query->payment_query_d + streamID,
            //                                                              database->log,
            //                                                              database->snapshot_d,
            //                                                              database->index_d);
            KERNEL_EXECUTE<<<EXECUTE_GRID_SIZE, EXECUTE_BLOCK_SIZE, 0, stream[epoch_ID]>>>(query->neworder_set_d + slotID,
                                                                                           query->payment_set_d + slotID,
                                                                                           query->neworder_query_d + slotID,
                                                                                           query->payment_query_d + slotID,
                                                                                           database->log,
                                                                                           database->snapshot_d,
                                                                                           database->index_d);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
                break;
            }
            // cudaDeviceSynchronize();
            cudaStreamSynchronize(stream[epoch_ID]);

            long long end_0 = current_time();
            // KERNEL_CHECK<<<CHECK_GRID_SIZE, CHECK_BLOCK_SIZE, 0>>>(query->neworder_set_d + streamID,
            //                                                        query->payment_set_d + streamID,
            //                                                        query->neworder_query_d + streamID,
            //                                                        query->payment_query_d + streamID,
            //                                                        database->log,
            //                                                        database->snapshot_d,
            //                                                        database->index_d);
            KERNEL_CHECK<<<CHECK_GRID_SIZE, CHECK_BLOCK_SIZE, 0, stream[epoch_ID]>>>(query->neworder_set_d + slotID,
                                                                                     query->payment_set_d + slotID,
                                                                                     query->neworder_query_d + slotID,
                                                                                     query->payment_query_d + slotID,
                                                                                     database->log,
                                                                                     database->snapshot_d,
                                                                                     database->index_d);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
                break;
            }
            // cudaDeviceSynchronize();
            cudaStreamSynchronize(stream[epoch_ID]);
            long long end_1 = current_time();
            // KERNEL_WRITEBACK<<<WRITEBACK_GRID_SIZE, WRITEBACK_BLOCK_SIZE, 0>>>(query->neworder_set_d + streamID,
            //                                                                    query->payment_set_d + streamID,
            //                                                                    query->neworder_query_d + streamID,
            //                                                                    query->payment_query_d + streamID,
            //                                                                    database->log,
            //                                                                    database->snapshot_d,
            //                                                                    database->index_d);
            KERNEL_WRITEBACK<<<WRITEBACK_GRID_SIZE, WRITEBACK_BLOCK_SIZE, 0, stream[epoch_ID]>>>(query->neworder_set_d + slotID,
                                                                                                 query->payment_set_d + slotID,
                                                                                                 query->neworder_query_d + slotID,
                                                                                                 query->payment_query_d + slotID,
                                                                                                 database->log,
                                                                                                 database->snapshot_d,
                                                                                                 database->index_d);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
                break;
            }
            // cudaDeviceSynchronize();
            cudaStreamSynchronize(stream[epoch_ID]);

            statistic_query(epoch_ID, query, stream);
            long long end_t = current_time();

            database->clear_LOG();

            float kernel_0_time = duration(start_kernel_0_t, end_t);
            float time_0 = duration(start_kernel_0_t, end_0);
            float time_1 = duration(end_0, end_1);
            float time_2 = duration(end_1, end_t);
            // std::cout << "Epoch " << epoch_ID << " kernel cost [" << kernel_0_time << " s].\n";
            // std::cout << "kernel 0 cost [" << time_0 << " s].\n";
            // std::cout << "kernel 1 cost [" << time_1 << " s].\n";
            // std::cout << "kernel 2 cost [" << time_2 << " s].\n";
            // check(database->log);
            if (epoch_ID >= WARMUP_TP && epoch_ID < EPOCH_TP - WARMUP_TP)
            {
                query->kernel_0_time_all += kernel_0_time;
                query->time_0_all += time_0;
                query->time_1_all += time_1;
                query->time_2_all += time_2;
            }

            cudaDeviceSynchronize();
        }

        for (size_t i = 0; i < STREAM_SIZE; i++)
        {
            cudaStreamDestroy(stream[i]);
        }

        // cudaEventDestroy(start_event);
        // cudaEventDestroy(end_event);
        // database->print();
        // query->print();
        float average_commit = ((float)query->commit_neworder + query->commit_payment) / (EPOCH_TP - 2 * WARMUP_TP);
        float average_n = ((float)query->commit_neworder) / (EPOCH_TP - 2 * WARMUP_TP);
        float average_p = ((float)query->commit_payment) / (EPOCH_TP - 2 * WARMUP_TP);
        float average_kernel_time = query->kernel_0_time_all / (EPOCH_TP - 2 * WARMUP_TP);
        float average_time_0 = query->time_0_all / (EPOCH_TP - 2 * WARMUP_TP);
        float average_time_1 = query->time_1_all / (EPOCH_TP - 2 * WARMUP_TP);
        float average_time_2 = query->time_2_all / (EPOCH_TP - 2 * WARMUP_TP);
        float average_copy_to_device = query->copy_to_device_cost / (EPOCH_TP - 2 * WARMUP_TP);
        float average_copy_to_host = query->copy_to_host_cost / (EPOCH_TP - 2 * WARMUP_TP);
        std::cout << "Kernel all cost [" << average_kernel_time << " s] in average.\n";
        std::cout << "Kernel 0 cost [" << average_time_0 << " s] in average.\n";
        std::cout << "Kernel 1 cost [" << average_time_1 << " s] in average.\n";
        std::cout << "Kernel 2 cost [" << average_time_2 << " s] in average.\n";
        std::cout << "copt to device cost [" << average_copy_to_device << " s] in average.\n";
        std::cout << "copy to host cost [" << average_copy_to_host << " s] in average.\n";
        std::cout << "memory copy speed is [" << query->memory_speed / (EPOCH_TP - 2 * WARMUP_TP) << " GB/s].\n";
        std::cout << "Commit [" << average_commit << "] in average.\n";
        std::cout << "Commit neworder [" << average_n << "] in average.\n";
        std::cout << "Commit payment [" << average_p << "] in average.\n";
        std::cout << "TPS [" << average_commit / average_kernel_time << "].\n";
        std::cout << "The warehouse size is [" << WAREHOUSE_SIZE << "].";
        std::cout << "The batch size is [" << BATCH_SIZE << "].\n\n";
        // std::cout << (WAREHOUSE_SIZE + DISTRICT_SIZE) * 2 * sizeof(unsigned int) * 31 << std::endl;
        free(database);
        free(query);
    }
    return 0;
}
