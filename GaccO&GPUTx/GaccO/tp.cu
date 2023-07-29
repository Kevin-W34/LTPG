#include <bits/stdc++.h>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Database.h"
#include "Query.h"
#include "Execute.h"
__global__ void KERNEL_CID(PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
{
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
{
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
{
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
{
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

__global__ void KERNEL_NEWORDER_ACCESS(NEWORDER_QUERY *neworder_query,
                                       NEWORDERQUERY_ACCESS *neworderquery_access,
                                       NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    GACCO_NEWORDER_NAMESPACE::get_Auxiliary_Structure(thID % NEWORDER_CNT, neworder_query, neworderquery_access, neworderquery_auxiliary);
}
__global__ void KERNEL_PAYMENT_ACCESS(PAYMENT_QUERY *payment_query,
                                      PAYMENTQUERY_ACCESS *paymentqueery_access,
                                      PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    GACCO_PAYMENT_NAMESPACE::get_Auxiliary_Structure(thID % PAYMENT_CNT, payment_query, paymentqueery_access, paymentquery_auxiliary);
}
void SORT_NEWORDER_ACCESS(Query *query)
{
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->warehouse_access.W_ID, query->neworderquery_access_d->warehouse_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->district_access.D_ID, query->neworderquery_access_d->district_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->customer_access.C_ID, query->neworderquery_access_d->customer_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->neworder_access.N_ID, query->neworderquery_access_d->neworder_access.TID, NEWORDER_CNT);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->order_access.O_ID, query->neworderquery_access_d->order_access.TID, NEWORDER_CNT);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->item_access.I_ID, query->neworderquery_access_d->item_access.TID, NEWORDER_CNT * 15);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->stock_access.S_ID, query->neworderquery_access_d->stock_access.TID, NEWORDER_CNT * 15);
    _sort_unsigned_int_up_down_id(query->neworderquery_access_d->orderline_access.OL_ID, query->neworderquery_access_d->orderline_access.TID, NEWORDER_CNT * 15);
}
void SORT_PAYMENT_ACCESS(Query *query)
{
    _sort_unsigned_int_up_down_id(query->paymentquery_access_d->warehouse_access.W_ID, query->paymentquery_access_d->warehouse_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->paymentquery_access_d->district_access.D_ID, query->paymentquery_access_d->district_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->paymentquery_access_d->customer_access.C_ID, query->paymentquery_access_d->customer_access.TID, NEWORDER_CNT + 2 * PAYMENT_CNT);
    _sort_unsigned_int_up_down_id(query->paymentquery_access_d->history_access.H_ID, query->paymentquery_access_d->history_access.TID, PAYMENT_CNT);
}
__global__ void KERNEL_NEWORDER_PREFIX(NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary)
{
    GACCO_NEWORDER_NAMESPACE::calculate_prefix_offset(neworderquery_auxiliary);
}
__global__ void KERNEL_PAYMENT_PREFIX(PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary)
{
    GACCO_PAYMENT_NAMESPACE::calculate_prefix_offset(paymentquery_auxiliary);
}
__global__ void KERNEL_NEWORDER_EXECUTE(NEWORDER_SET *neworder_set,
                                        NEWORDER_QUERY *neworder_query,
                                        NEWORDERQUERY_ACCESS *neworderquery_access,
                                        NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary,
                                        SNAPSHOT *snapshot)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    GACCO_NEWORDER_NAMESPACE::execute(thID, neworder_set, neworder_query, neworderquery_access, neworderquery_auxiliary, snapshot);
}
__global__ void KERNEL_PAYMENT_EXECUTE(PAYMENT_SET *payment_set,
                                       PAYMENT_QUERY *payment_query,
                                       PAYMENTQUERY_ACCESS *paymentqueery_access,
                                       PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary,
                                       SNAPSHOT *snapshot)
{
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    GACCO_PAYMENT_NAMESPACE::execute(thID, payment_set, payment_query, paymentqueery_access, paymentquery_auxiliary, snapshot);
}
int main(int argc, char const *argv[])
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
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        // query->random_choose_query(epoch_ID, stream);
        cudaError_t err;
        long long start_kernel_0_t = current_time();
        cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        query->random_choose_query(epoch_ID, stream);

        // KERNEL_CID<<<GRID_SIZE, BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                          query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                          database->log,
        //                                                                          database->snapshot_d,
        //   database->index_d);
        
        KERNEL_NEWORDER_ACCESS<<<8 * NEWORDER_CNT / 512 + 1, 512>>>(query->neworder_query_d,
                                                                    query->neworderquery_access_d,
                                                                    query->neworderquery_auxiliary_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);

        // KERNEL_EXECUTE<<<EXECUTE_GRID_SIZE, EXECUTE_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                              database->log,
        //                                                                                              database->snapshot_d,
        //                                                                                              database->index_d);

        SORT_NEWORDER_ACCESS(query);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        long long end_0 = current_time();
        // KERNEL_CHECK<<<CHECK_GRID_SIZE, CHECK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                        database->log,
        //                                                                                        database->snapshot_d,
        //                                                                                        database->index_d);
        KERNEL_NEWORDER_PREFIX<<<8, 512>>>(query->neworderquery_auxiliary_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        long long end_1 = current_time();
        // KERNEL_WRITEBACK<<<WRITEBACK_GRID_SIZE + 1, WRITEBACK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        database->log,
        //                                                                                                        database->snapshot_d,
        //                                                                                                        database->index_d);
        KERNEL_NEWORDER_EXECUTE<<<ORDERLINE_SIZE / 512 + 1, 512>>>(query->neworder_set_d,
                                                                   query->neworder_query_d,
                                                                   query->neworderquery_access_d,
                                                                   query->neworderquery_auxiliary_d,
                                                                   database->snapshot_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        
        long long end_2 = current_time();
        KERNEL_PAYMENT_ACCESS<<<4 * PAYMENT_CNT / 512 + 1, 512>>>(query->payment_query_d,
                                                                  query->paymentquery_access_d,
                                                                  query->paymentquery_auxiliary_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);

        // KERNEL_EXECUTE<<<EXECUTE_GRID_SIZE, EXECUTE_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                              query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                              database->log,
        //                                                                                              database->snapshot_d,
        //                                                                                              database->index_d);

        SORT_PAYMENT_ACCESS(query);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        // long long end_2 = current_time();
        // KERNEL_CHECK<<<CHECK_GRID_SIZE, CHECK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                        query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                        database->log,
        //                                                                                        database->snapshot_d,
        //                                                                                        database->index_d);
        KERNEL_PAYMENT_PREFIX<<<4, 512>>>(query->paymentquery_auxiliary_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        long long end_3 = current_time();
        // KERNEL_WRITEBACK<<<WRITEBACK_GRID_SIZE + 1, WRITEBACK_BLOCK_SIZE, 0, stream[epoch_ID % STREAM_SIZE]>>>(query->neworder_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->payment_set_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->neworder_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        query->payment_query_d + epoch_ID % STREAM_SIZE,
        //                                                                                                        database->log,
        //                                                                                                        database->snapshot_d,
        //                                                                                                        database->index_d);
        KERNEL_PAYMENT_EXECUTE<<<CUSTOMER_SIZE / 512 + 1, 512>>>(query->payment_set_d,
                                                                 query->payment_query_d,
                                                                 query->paymentquery_access_d,
                                                                 query->paymentquery_auxiliary_d,
                                                                 database->snapshot_d);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
            break;
        }
        cudaDeviceSynchronize();
        // cudaStreamSynchronize(stream[epoch_ID % STREAM_SIZE]);
        // if (epoch_ID >= WARMUP_TP && epoch_ID < EPOCH_TP - WARMUP_TP)
        // {
        
        statistic_query(epoch_ID, query, stream);
        // }
        long long end_t = current_time();
        float kernel_0_time = duration(start_kernel_0_t, end_t);
        float time_0 = duration(start_kernel_0_t, end_0);
        float time_1 = duration(end_0, end_1);
        float time_2 = duration(end_1, end_2);
        float time_3 = duration(end_2, end_3);
        float time_4 = duration(end_3, end_t);
        std::cout << "Epoch " << epoch_ID << " kernel cost [" << kernel_0_time << " s].\n";
        std::cout << "kernel 0 cost [" << time_0 << " s].\n";
        std::cout << "kernel 1 cost [" << time_1 << " s].\n";
        std::cout << "kernel 2 cost [" << time_2 << " s].\n";
        std::cout << "kernel 3 cost [" << time_3 << " s].\n";
        std::cout << "kernel 4 cost [" << time_4 << " s].\n";
        // check(database->log);
        if (epoch_ID >= WARMUP_TP && epoch_ID < EPOCH_TP - WARMUP_TP)
        {
            query->kernel_0_time_all += kernel_0_time;
            query->time_0_all += time_0;
            query->time_1_all += time_1;
            query->time_2_all += time_2;
            query->time_3_all += time_3;
            query->time_4_all += time_4;
            // statistic_query(epoch_ID, query);
        }
        database->clear_LOG();
        query->clear_auxliary();
    }
    for (size_t i = 0; i < STREAM_SIZE; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    database->print();
    query->print();
    float average_commit = ((float)query->commit_all) / (EPOCH_TP - 2 * WARMUP_TP);
    float average_n = ((float)query->commit_neworder) / (EPOCH_TP - 2 * WARMUP_TP);
    float average_p = ((float)query->commit_payment) / (EPOCH_TP - 2 * WARMUP_TP);
    float average_kernel_time = query->kernel_0_time_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_0 = query->time_0_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_1 = query->time_1_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_2 = query->time_2_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_3 = query->time_3_all / (EPOCH_TP - 2 * WARMUP_TP);
    float average_time_4 = query->time_4_all / (EPOCH_TP - 2 * WARMUP_TP);
    std::cout << "Kernel all cost [" << average_kernel_time << " s] in average.\n\n";
    std::cout << "Kernel 0 cost [" << average_time_0 << " s] in average.\n";
    std::cout << "Kernel 1 cost [" << average_time_1 << " s] in average.\n";
    std::cout << "Kernel 2 cost [" << average_time_2 << " s] in average.\n";
    std::cout << "Kernel 3 cost [" << average_time_3 << " s] in average.\n";
    std::cout << "Kernel 4 cost [" << average_time_4 << " s] in average.\n\n";
    std::cout << "memory copy speed is [" << query->memory_speed / (EPOCH_TP - 2 * WARMUP_TP) << " GB/s].\n\n";
    std::cout << "Commit [" << average_commit << "] in average.\n\n";
    std::cout << "Commit neworder [" << average_n << "] in average.\n\n";
    std::cout << "Commit payment [" << average_p << "] in average.\n\n";
    std::cout << "TPS [" << average_commit / average_kernel_time << "].\n\n";
    std::cout << "The warehouse size is [" << WAREHOUSE_SIZE << "].";
    std::cout << "The batch size is [" << BATCH_SIZE << "].\n\n";
    free(database);
    free(query);
    return 0;
}
