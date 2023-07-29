#pragma once

#include <time.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "Predefine.h"
#include "Datastructure.h"

long long current_time()
{ // 获取当前时间
    timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    long long time_t = time.tv_sec * 1000000 + time.tv_nsec / 1000;
    return time_t;
}
float duration(long long start_t, long long end_t)
{ // 计算时长
    float time = ((float)(end_t - start_t)) / 1000000.0;
    return time;
}
__device__ unsigned int __FIND_CID_INDEX(unsigned int C_LAST,              // C_LAST
                                         unsigned int distance,            // 返回值
                                         unsigned int *C_NAME_INDEX_START, // 起始位置
                                         unsigned int C_NAME_INDEX_LENGTH, // 数组长度
                                         unsigned int *snapshot)
{ // 通过 C_NAME_INDEX 获取其在 snapshot 中的位置, 进而获得对应顾客的 C_ID
    unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int C_ID = 0;
    unsigned int c_last_id = C_LAST & 0xff;
    unsigned int c_last = C_LAST >> 8;
    unsigned int count = 0;
    unsigned int cumulative = 0;
    unsigned int flag = 0;
    for (size_t i = thID % WARP_SIZE; i < C_NAME_INDEX_LENGTH; i += WARP_SIZE)
    {
        unsigned int C_LAST = C_NAME_INDEX_START[distance + i];
        unsigned int tmp = (C_LAST == c_last);
        for (size_t ii = 0; ii < WARP_SIZE; ii++)
        {
            count += __shfl_sync(0xffffffff, tmp, i);
        }
        cumulative += count;
        count = 0;
        flag += WARP_SIZE;
        if (cumulative >= c_last_id)
        {
            break;
        }
    }
    unsigned int result = 0;
    if (thID % WARP_SIZE == 0 && cumulative >= c_last_id)
    {
        for (size_t i = flag - 1; i > 0; i--)
        {
            if (C_NAME_INDEX_START[distance + i] == c_last)
            {
                if (cumulative == c_last_id)
                {
                    result = distance + i;
                    break;
                }
                cumulative--;
            }
        }
        C_ID = snapshot[0 * CUSTOMER_SIZE + result];
        // printf("C_ID %d, C_LAST 0x%08x\n", C_ID, C_LAST);
    }
    return C_ID;
}
__device__ void __READ(unsigned int NO,
                       unsigned int OP,
                       unsigned int column,
                       unsigned int Loc,
                       unsigned int query_cnt,
                       unsigned long long table_size,
                       unsigned int column_size,
                       int new_value,
                       unsigned int *set_local_set,
                       unsigned int *table)
{
    unsigned int index = OP * 32 * query_cnt + column * query_cnt + NO;
    unsigned int index_t = column * table_size + Loc;
    set_local_set[index] = __ldg(&table[index_t]);
}
__device__ void __WRITE(unsigned int NO,
                        unsigned int OP,
                        unsigned int column,
                        unsigned int Loc,
                        unsigned int query_cnt,
                        unsigned long long table_size,
                        unsigned int column_size,
                        int new_value,
                        unsigned int *set_local_set,
                        unsigned int *table)
{
    unsigned int index = OP * 32 * query_cnt + column * query_cnt + NO;
    unsigned int index_t = column * table_size + Loc;
    set_local_set[index] = __ldg(&table[index_t]);
}
__device__ void __REGISTER(unsigned int isFrequent,
                           unsigned int NO,        // NO 表示当前实例是这类事务实例的第 NO 个
                           unsigned int OP,        // OP 表示当前操作是本示例中的第 OP 个
                           unsigned int query_cnt, // query_cnt表示当前事务个数, 如 NEWORDER_OP_CNT, PAYMENT_OP_CNT 等
                           unsigned int TID,
                           unsigned int Loc,
                           unsigned int log_loc_up,   // Loc 表示当前访问的数据在第几行
                           unsigned int log_loc_down, // Loc 表示当前访问的数据在第几行
                           LOG *log)
{
    // unsigned int index = NO * query_cnt + OP;
    // LOG_LOC_UP[index] = log_loc_up;
    // LOG_LOC_DOWN[index] = log_loc_down;
    // LOG_TID[index] = TID;
    unsigned int table_id = log_loc_up >> 24;
    unsigned int op_type = log_loc_down & 0x1;
    if (table_id == WAREHOUSE_ID)
    {
        if (isFrequent == 1)
        {
            // printf("TID %d, table_id %d, Loc %d, op_type %d,\n", TID, table_id, Loc, op_type);
            if (op_type == READ_TYPE)
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_WAREHOUSE_R_index, 1);
                // atomicAdd(&log->LOG_WAREHOUSE_R[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_WAREHOUSE_R[index], Loc);
                atomicMin(&log->LOG_WAREHOUSE_R[WAREHOUSE_SIZE + Loc], TID);
            }
            else
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_WAREHOUSE_W_index, 1);
                // atomicAdd(&log->LOG_WAREHOUSE_W[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_WAREHOUSE_W[index], Loc);
                atomicMin(&log->LOG_WAREHOUSE_W[WAREHOUSE_SIZE + Loc], TID);
                // printf("TID %d, table_id %d, Loc %d, op_type %d\n", TID, table_id, Loc, op_type);
            }
        }
        else
        {
            // printf("TID %d, table_id %d, Loc %d, op_type %d,\n", TID, table_id, Loc, op_type);
            if (op_type == READ_TYPE)
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_WAREHOUSE_R_index, 1);
                // atomicAdd(&log->LOG_WAREHOUSE_R[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_WAREHOUSE_R[index], Loc);
                atomicMin(&log->LOG_WAREHOUSE_R[Loc], TID);
            }
            else
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_WAREHOUSE_W_index, 1);
                // atomicAdd(&log->LOG_WAREHOUSE_W[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_WAREHOUSE_W[index], Loc);
                atomicMin(&log->LOG_WAREHOUSE_W[Loc], TID);
                // printf("TID %d, table_id %d, Loc %d, op_type %d\n", TID, table_id, Loc, op_type);
            }
        }
    }
    else if (table_id == DISTRICT_ID)
    {
        if (isFrequent == 1)
        {
            if (op_type == READ_TYPE)
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_DISTRICT_R_index, 1);
                // atomicAdd(&log->LOG_DISTRICT_R[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_DISTRICT_R[index], Loc);
                atomicMin(&log->LOG_DISTRICT_R[DISTRICT_SIZE + Loc], TID);
            }
            else
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_DISTRICT_W_index, 1);
                // atomicAdd(&log->LOG_DISTRICT_W[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_DISTRICT_W[index], Loc);
                atomicMin(&log->LOG_DISTRICT_W[DISTRICT_SIZE + Loc], TID);
            }
        }
        else
        {
            if (op_type == READ_TYPE)
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_DISTRICT_R_index, 1);
                // atomicAdd(&log->LOG_DISTRICT_R[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_DISTRICT_R[index], Loc);
                atomicMin(&log->LOG_DISTRICT_R[Loc], TID);
            }
            else
            {
                // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_DISTRICT_W_index, 1);
                // atomicAdd(&log->LOG_DISTRICT_W[Loc], 1);
                // atomicExch(&log->mini_log[miniID].LOG_DISTRICT_W[index], Loc);
                atomicMin(&log->LOG_DISTRICT_W[Loc], TID);
            }
        }
    }
    else if (table_id == CUSTOMER_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_CUSTOMER_R_index, 1);
            // atomicAdd(&log->LOG_CUSTOMER_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_CUSTOMER_R[index], Loc);
            atomicMin(&log->LOG_CUSTOMER_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_CUSTOMER_W_index, 1);
            // atomicAdd(&log->LOG_CUSTOMER_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_CUSTOMER_W[index], Loc);
            atomicMin(&log->LOG_CUSTOMER_W[Loc], TID);
        }
    }
    else if (table_id == HISTORY_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_HISTORY_R_index, 1);
            // atomicAdd(&log->LOG_HISTORY_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_HISTORY_R[index], Loc);
            atomicMin(&log->LOG_HISTORY_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_HISTORY_W_index, 1);
            // atomicAdd(&log->LOG_HISTORY_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_HISTORY_W[index], Loc);
            atomicMin(&log->LOG_HISTORY_W[Loc], TID);
        }
    }
    else if (table_id == NEWORDER_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_NEWORDER_R_index, 1);
            // atomicAdd(&log->LOG_NEWORDER_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_NEWORDER_R[index], Loc);
            atomicMin(&log->LOG_NEWORDER_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_NEWORDER_W_index, 1);
            // atomicAdd(&log->LOG_NEWORDER_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_NEWORDER_W[index], Loc);
            atomicMin(&log->LOG_NEWORDER_W[Loc], TID);
        }
    }
    else if (table_id == ORDER_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ORDER_R_index, 1);
            // atomicAdd(&log->LOG_ORDER_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ORDER_R[index], Loc);
            atomicMin(&log->LOG_ORDER_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ORDER_W_index, 1);
            // atomicAdd(&log->LOG_ORDER_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ORDER_W[index], Loc);
            atomicMin(&log->LOG_ORDER_W[Loc], TID);
        }
    }
    else if (table_id == ORDERLINE_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ORDERLINE_R_index, 1);
            // atomicAdd(&log->LOG_ORDERLINE_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ORDERLINE_R[index], Loc);
            atomicMin(&log->LOG_ORDERLINE_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ORDERLINE_W_index, 1);
            // atomicAdd(&log->LOG_ORDERLINE_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ORDERLINE_W[index], Loc);
            atomicMin(&log->LOG_ORDERLINE_W[Loc], TID);
        }
    }
    else if (table_id == STOCK_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_STOCK_R_index, 1);
            // atomicAdd(&log->LOG_STOCK_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_STOCK_R[index], Loc);
            atomicMin(&log->LOG_STOCK_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_STOCK_W_index, 1);
            // atomicAdd(&log->LOG_STOCK_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_STOCK_W[index], Loc);
            atomicMin(&log->LOG_STOCK_W[Loc], TID);
        }
    }
    else if (table_id == ITEM_ID)
    {
        if (op_type == READ_TYPE)
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ITEM_R_index, 1);
            // atomicAdd(&log->LOG_ITEM_R[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ITEM_R[index], Loc);
            atomicMin(&log->LOG_ITEM_R[Loc], TID);
        }
        else
        {
            // unsigned int index = atomicAdd(&log->mini_log[miniID].LOG_ITEM_W_index, 1);
            // atomicAdd(&log->LOG_ITEM_W[Loc], 1);
            // atomicExch(&log->mini_log[miniID].LOG_ITEM_W[index], Loc);
            atomicMin(&log->LOG_ITEM_W[Loc], TID);
        }
    }
    // printf("register miniID = %d, NO = %d, TID = %d, log_loc_up = 0x%08x, log_loc_down = 0x%08x\n", miniID, NO, TID, log_loc_up, log_loc_down);
    // printf("register miniID = %d, NO = %d, LOG_TID[index] = %d, LOG_LOC_UP[index] = 0x%08x, LOG_LOC_DOWN[index] = 0x%08x\n", miniID, NO, LOG_TID[index], LOG_LOC_UP[index], LOG_LOC_DOWN[index]);
    // printf("TID %d, table_id %d, Loc %d, op_type %d,\n", TID, table_id, Loc, op_type);
}
__device__ void __CHECK(unsigned int isFrequent,
                        unsigned int NO,
                        unsigned int Loc,
                        unsigned int TID,
                        unsigned int log_loc_up,
                        unsigned int log_loc_down,
                        unsigned int *CONFLICT_MARK,
                        LOG *log)
{
    unsigned int table_id = log_loc_up >> 24;
    unsigned int op_type = log_loc_down & 0x1;
    // printf("table_id %d, op_type %d\n", table_id, op_type);
    unsigned int existedTID = 0xffffffff;
    if (table_id == WAREHOUSE_ID)
    {
        if (isFrequent == 1)
        {
            // if (op_type == READ_TYPE)
            // {
            //     existedTID = __ldg(&log->LOG_WAREHOUSE_R[WAREHOUSE_SIZE + Loc]);
            // }
            // else
            // {
            //     existedTID = __ldg(&log->LOG_WAREHOUSE_W[WAREHOUSE_SIZE + Loc]);
            //     // printf("TID %d, existedTID %d, table_id %d, Loc %d, op_type %d\n", TID, existedTID, table_id, Loc, op_type);
            // }
        }
        else
        {
            if (op_type == READ_TYPE)
            {
                existedTID = __ldg(&log->LOG_WAREHOUSE_R[Loc]);
            }
            else
            {
                existedTID = __ldg(&log->LOG_WAREHOUSE_W[Loc]);
                // printf("TID %d, existedTID %d, table_id %d, Loc %d, op_type %d\n", TID, existedTID, table_id, Loc, op_type);
            }
        }
    }
    else if (table_id == DISTRICT_ID)
    {
        if (isFrequent == 1)
        {
            // if (op_type == READ_TYPE)
            // {
            //     existedTID = __ldg(&log->LOG_DISTRICT_R[DISTRICT_SIZE + Loc]);
            // }
            // else
            // {
            //     existedTID = __ldg(&log->LOG_DISTRICT_W[DISTRICT_SIZE + Loc]);
            // }
        }
        else
        {
            if (op_type == READ_TYPE)
            {
                existedTID = __ldg(&log->LOG_DISTRICT_R[Loc]);
            }
            else
            {
                existedTID = __ldg(&log->LOG_DISTRICT_W[Loc]);
            }
        }
    }
    else if (table_id == CUSTOMER_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_CUSTOMER_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_CUSTOMER_W[Loc]);
        }
    }
    else if (table_id == HISTORY_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_HISTORY_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_HISTORY_W[Loc]);
        }
    }
    else if (table_id == NEWORDER_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_NEWORDER_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_NEWORDER_W[Loc]);
        }
    }
    else if (table_id == ORDER_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_ORDER_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_ORDER_W[Loc]);
        }
    }
    else if (table_id == ORDERLINE_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_ORDERLINE_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_ORDERLINE_W[Loc]);
        }
    }
    else if (table_id == STOCK_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_STOCK_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_STOCK_W[Loc]);
        }
    }
    else if (table_id == ITEM_ID)
    {
        if (op_type == READ_TYPE)
        {
            existedTID = __ldg(&log->LOG_ITEM_R[Loc]);
        }
        else
        {
            existedTID = __ldg(&log->LOG_ITEM_W[Loc]);
        }
    }
    if (TID > existedTID)
    {
        atomicAdd(&CONFLICT_MARK[NO], 1);
        // printf("table_id %d, op_type %d, Loc %d\n", table_id, op_type, Loc);
    }
}

__device__ void __WRITEBACK(unsigned int NO,
                            unsigned int OP,
                            unsigned int column,
                            unsigned int Loc,
                            unsigned int query_cnt,
                            unsigned long long table_size,
                            unsigned int log_loc_up,
                            unsigned int isCommit,
                            unsigned int *set_local_set,
                            unsigned int *table)
{
    // printf("origin %d, write %d\n", table[column* column_size + Loc], set_local_set[OP*32*query_cnt+column*32+NO]);
    unsigned int index = OP * 32 * query_cnt + column * query_cnt + NO;
    unsigned int index_t = column * table_size + Loc;
    table[index_t] = __ldg(&set_local_set[index]);
}
__device__ void __REDUCE_UPGRADE(unsigned int NO,
                                 unsigned int OP,
                                 unsigned int column,
                                 unsigned int Loc,
                                 unsigned int query_cnt,
                                 unsigned long long isCommit,
                                 unsigned int table_id,
                                 LOG *log,
                                 unsigned int *set_local_set,
                                 unsigned int *table)
{
    const unsigned int laneId = threadIdx.x % WARP_SIZE;
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    __shared__ unsigned int warpLevelSum[16];
    if (blockIdx.x < WAREHOUSE_SIZE)
    {
        unsigned int warehouse_id = blockIdx.x;
        for (size_t j = 0; j < (PAYMENT_CNT / WRITEBACK_BLOCK_SIZE == 0 ? 1 : PAYMENT_CNT / WRITEBACK_BLOCK_SIZE); j++)
        {
            unsigned int YTD = 0;
            if (threadIdx.x + j * WRITEBACK_BLOCK_SIZE < PAYMENT_CNT && isCommit == 0)
            {
                YTD = (Loc == warehouse_id ? log->W_YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] : 0);
            }
            YTD += __shfl_up_sync(0xffffffff, YTD, 16);
            YTD += __shfl_up_sync(0xffffffff, YTD, 8);
            YTD += __shfl_up_sync(0xffffffff, YTD, 4);
            YTD += __shfl_up_sync(0xffffffff, YTD, 2);
            YTD += __shfl_up_sync(0xffffffff, YTD, 1);
            if (laneId == 31)
            {
                warpLevelSum[warpId] = YTD;
            }
            __syncthreads();
            unsigned int YTD_sum = laneId < 16 ? warpLevelSum[laneId] : 0;
            for (size_t i = 0; i < warpId; i++)
            {
                YTD += __shfl_sync(0xffffffff, YTD_sum, i);
            }
            for (size_t i = 0; i < j; i++)
            {
                YTD += log->TMP_W_YTD[warehouse_id];
            }
            if (Loc == warehouse_id && threadIdx.x + j * WRITEBACK_BLOCK_SIZE < PAYMENT_CNT)
            {
                log->W_YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] = YTD;
                atomicMax(&log->TMP_W_YTD[warehouse_id], YTD);
                // printf("%d, %d, %d\n", NO, Loc, YTD);
            }
        }
    }
    if (blockIdx.x < DISTRICT_SIZE)
    {
        unsigned int district_id = blockIdx.x;
        for (size_t j = 0; j < (PAYMENT_CNT / WRITEBACK_BLOCK_SIZE == 0 ? 1 : PAYMENT_CNT / WRITEBACK_BLOCK_SIZE); j++)
        {
            unsigned int YTD = 0;
            if (threadIdx.x + j * WRITEBACK_BLOCK_SIZE < PAYMENT_CNT && isCommit == 0)
            {
                YTD = (Loc == district_id ? log->D_YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] : 0);
            }
            YTD += __shfl_up_sync(0xffffffff, YTD, 16);
            YTD += __shfl_up_sync(0xffffffff, YTD, 8);
            YTD += __shfl_up_sync(0xffffffff, YTD, 4);
            YTD += __shfl_up_sync(0xffffffff, YTD, 2);
            YTD += __shfl_up_sync(0xffffffff, YTD, 1);
            if (laneId == 31)
            {
                warpLevelSum[warpId] = YTD;
            }
            __syncthreads();
            unsigned int YTD_sum = laneId < 16 ? warpLevelSum[laneId] : 0;
            for (size_t i = 0; i < warpId; i++)
            {
                YTD += __shfl_sync(0xffffffff, YTD_sum, i);
            }
            for (size_t i = 0; i < j; i++)
            {
                YTD += log->TMP_D_YTD[district_id];
            }
            if (Loc == district_id && threadIdx.x + j * WRITEBACK_BLOCK_SIZE < PAYMENT_CNT)
            {
                log->D_YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] = YTD;
                atomicMax(&log->TMP_D_YTD[district_id], YTD);
                // printf("%d, %d, %d\n", NO, Loc, YTD);
            }
        }
    }
}

void check(LOG *log)
{
    unsigned int all_count = 0;
    unsigned int n_count = 0;
    unsigned int p_count = 0;

    for (int i = 0; i < MINI_BATCH_CNT; i++)
    {
        // printf("log_tid = 0x%08x, log_loc_up = 0x%08x, log_loc_down = 0x%08x\n", log->LOG_TID[i], log->LOG_LOC_UP[i], log->LOG_LOC_DOWN[i]);
        // for (int j = 0; j < MINI_NEWORDER_CNT * NEWORDER_OP_CNT; j++)
        // {
        //     if ()
        //     {
        //         all_count += 1;
        //         n_count += 1;
        //     }
        // }
        // for (int j = 0; j < MINI_PAYMENT_CNT * PAYMENT_OP_CNT; j++)
        // {
        //     if ()
        //     {
        //         all_count += 1;
        //         p_count += 1;
        //     }
        // }
    }
    printf("all success %d\n", OP_CNT - all_count);
    printf("all fail %d\n", all_count);

    printf("n success = %d\n", NEWORDER_CNT * NEWORDER_OP_CNT - n_count);
    printf("n fail = %d\n", n_count);

    printf("p success %d\n", PAYMENT_CNT * PAYMENT_OP_CNT - p_count);
    printf("p fail %d\n", p_count);

    unsigned int les_count = 0;
    unsigned int lcs_count = 0;
    unsigned int lws_count = 0;
    printf("e success %d\n", MINI_BATCH_CNT - les_count);
    printf("e fail %d\n", les_count);
    printf("c success %d\n", MINI_BATCH_CNT - lcs_count);
    printf("c fail %d\n", lcs_count);
    printf("w success %d\n", MINI_BATCH_CNT - lws_count);
    printf("w fail %d\n", lws_count);
}

__device__ void _swap_unsigned_int_up_down_id(unsigned int *f1,
                                              unsigned int *f2,
                                              unsigned int *TID1,
                                              unsigned int *TID2)
{ // sort_by_key 的双调排序实现, 交换排序元素的同时交换对应 ID
    unsigned int tmp = *f1;
    *f1 = *f2;
    *f2 = tmp;
    tmp = *TID1;
    *TID1 = *TID2;
    *TID2 = tmp;
}
__global__ void _bitonic_sort_unsigned_int_up_down_id(unsigned int *d_arr,
                                                      unsigned int *d_TID,
                                                      unsigned int input_arr_len,
                                                      unsigned int stride,
                                                      unsigned int inner_stride)
{
    unsigned int thID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thID_other = thID ^ inner_stride; // 使元素仅比较一次
    if (thID < thID_other)
    {
        // 操纵左侧的半部分
        unsigned int d_arr_thID = d_arr[thID];
        unsigned int d_arr_thID_arr = d_arr[thID_other];
        if ((thID & stride) == 0) // 判断单增还是单减
        {
            // 此处将留升序
            if (d_arr_thID > d_arr_thID_arr)
            {
                _swap_unsigned_int_up_down_id(&d_arr[thID], &d_arr[thID_other], &d_TID[thID], &d_TID[thID_other]);
            }
        }
        else
        {
            // 此处将留降序
            if (d_arr_thID < d_arr_thID_arr)
            {
                _swap_unsigned_int_up_down_id(&d_arr[thID], &d_arr[thID_other], &d_TID[thID], &d_TID[thID_other]);
            }
        }
    }
}

void _sort_unsigned_int_up_down_id(unsigned int *arr,
                                   unsigned int *TID,
                                   unsigned int len)
{
    // 首先检查长度是否为 2 的幂
    unsigned int twoUpper = 1;
    for (; twoUpper < len; twoUpper <<= 1)
    {
        if (twoUpper == len)
        {
            break;
        }
    }
    // 如果是 host 指针，返回
    unsigned int *d_input_arr;
    unsigned int *d_input_TID;
    unsigned int input_arr_len;
    if (twoUpper == len)
    {
        d_input_arr = arr;
        d_input_TID = TID;
        input_arr_len = len;
    }
    else
    {
        // 需要 padding
        input_arr_len = twoUpper;
        cudaMalloc(&d_input_arr, sizeof(unsigned int) * input_arr_len);
        cudaMalloc(&d_input_TID, sizeof(unsigned int) * input_arr_len);
        // 然后初始化
        cudaMemcpy(d_input_arr, arr, sizeof(unsigned int) * len, cudaMemcpyDeviceToDevice);
        cudaMemset(d_input_arr + len, (unsigned int)0xffffffff, sizeof(unsigned int) * (input_arr_len - len));
        cudaMemcpy(d_input_TID, TID, sizeof(unsigned int) * len, cudaMemcpyDeviceToDevice);
        cudaMemset(d_input_TID + len, (unsigned int)0xffffffff, sizeof(unsigned int) * (input_arr_len - len));
    }
    // std::cout << input_arr_len << std::endl;
    dim3 grid_dim((input_arr_len / 512 == 0) ? 1 : input_arr_len / 512);
    dim3 block_dim((input_arr_len / 512 == 0) ? input_arr_len : 512);
    // 排序过程(重点)
    for (unsigned int stride = 2; stride <= input_arr_len; stride <<= 1)
    {
        for (unsigned int inner_stride = stride >> 1; inner_stride >= 1; inner_stride >>= 1)
        {
            _bitonic_sort_unsigned_int_up_down_id<<<grid_dim, block_dim>>>(d_input_arr, d_input_TID, input_arr_len, stride, inner_stride);
            cudaDeviceSynchronize();
        }
    }
    cudaDeviceSynchronize();
    // 如果 padding 过，则此处还原
    if (twoUpper != len)
    {
        cudaMemcpy(arr, d_input_arr, sizeof(unsigned int) * len, cudaMemcpyDeviceToDevice);
        cudaFree(d_input_arr);
        cudaMemcpy(TID, d_input_TID, sizeof(unsigned int) * len, cudaMemcpyDeviceToDevice);
        cudaFree(d_input_TID);
    }
}

__device__ void prefix_sum_in_block(unsigned int *d_in, unsigned int N)
{
    unsigned int thID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int laneId = threadIdx.x % warpSize;
    unsigned int warpId = threadIdx.x / warpSize;
    __shared__ unsigned int tmp[16];
    unsigned int sum = 0;
    if (thID < N)
    {
        sum = d_in[thID];
    }
    unsigned int shfl = 0;
    shfl = __shfl_up_sync(0xffffffff, sum, 1);
    if (laneId >= 1)
    {
        sum += shfl;
    }
    shfl = __shfl_up_sync(0xffffffff, sum, 2);
    if (laneId >= 2)
    {
        sum += shfl;
    }
    shfl = __shfl_up_sync(0xffffffff, sum, 4);
    if (laneId >= 4)
    {
        sum += shfl;
    }
    shfl = __shfl_up_sync(0xffffffff, sum, 8);
    if (laneId >= 8)
    {
        sum += shfl;
    }
    shfl = __shfl_up_sync(0xffffffff, sum, 16);
    if (laneId >= 16)
    {
        sum += shfl;
    }
    if (laneId == 31)
    {
        tmp[warpId] = sum;
        // printf("%d, %d\n", warpId, sum);
    }
    __syncthreads();
    unsigned int warpLevelSum = laneId < 16 ? tmp[laneId] : 0;
    // printf("%d, %d\n",laneId,warpLevelSum);
    for (size_t j = 0; j < warpId; j++)
    {
        sum += __shfl_sync(0xffffffff, warpLevelSum, j);
    }
    if (thID < N)
    {
        d_in[thID] = sum;
    }
}
__device__ void prefix_sum_between_block(unsigned int *d_in, unsigned int N)
{
    unsigned int thID = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned int laneId = threadIdx.x % warpSize;
    const unsigned int blockId = blockIdx.x;
    unsigned int sum = 0;
    if (thID < N)
    {
        sum = d_in[thID];
    }
    if (blockId > 0)
    {
        __shared__ unsigned int blockLevelSum[2];
        if (threadIdx.x < blockId)
        {
            blockLevelSum[threadIdx.x] = d_in[(threadIdx.x + 1) * 512 - 1];
        }
        __syncthreads();
        for (size_t i = 0; i < blockId; i++)
        {
            sum += blockLevelSum[i];
        }
    }
    if (thID < N)
    {
        d_in[thID] = sum;
    }
}
