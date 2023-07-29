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
namespace YCSB
{
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
        unsigned int index = OP * YCSB_COLUMN * query_cnt + column * query_cnt + NO;
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
        unsigned int index = OP * YCSB_COLUMN * query_cnt + column * query_cnt + NO;
        unsigned int index_t = column * table_size + Loc;
        set_local_set[index] = __ldg(&table[index_t]);
    }
    __device__ void __REGISTER(unsigned int miniID,
                               unsigned int NO,        // NO 表示当前实例是这类事务实例的第 NO 个
                               unsigned int OP,        // OP 表示当前操作是本示例中的第 OP 个
                               unsigned int query_cnt, // query_cnt表示当前事务个数, 如 NEWORDER_OP_CNT, PAYMENT_OP_CNT 等
                               unsigned int TID,
                               unsigned int Loc,
                               unsigned int log_loc_up,   // Loc 表示当前访问的数据在第几行
                               unsigned int log_loc_down, // Loc 表示当前访问的数据在第几行
                               YCSB_LOG *log)
    {
        unsigned int op_type = log_loc_down & 0x1;
        if (op_type == READ_TYPE)
        {
            if (DATA_DISTRIBUTION == ZIPFIAN)
            {
                if (Loc != 0)
                {
                    atomicMin(&log->YCSB_R[Loc], TID);
                }
            }
            else
            {
                atomicMin(&log->YCSB_R[Loc], TID);
            }
        }
        else
        {
            if (DATA_DISTRIBUTION == ZIPFIAN)
            {
                if (Loc != 0)
                {
                    atomicMin(&log->YCSB_W[Loc], TID);
                }
            }
            else
            {
                atomicMin(&log->YCSB_W[Loc], TID);
            }
        }
    }
    __device__ void __CHECK(unsigned int miniID,
                            unsigned int NO,
                            unsigned int Loc,
                            unsigned int TID,
                            unsigned int log_loc_up,
                            unsigned int log_loc_down,
                            unsigned int *CONFLICT_MARK,
                            YCSB_LOG *log)
    {
        unsigned int op_type = log_loc_down & 0x1;
        unsigned int existedTID = 0xffffffff;
        if (op_type == READ_TYPE)
        {
            if (DATA_DISTRIBUTION == ZIPFIAN)
            {
                if (Loc != 0)
                {
                    existedTID = __ldg(&log->YCSB_R[Loc]);
                }
            }
            else
            {
                existedTID = __ldg(&log->YCSB_R[Loc]);
            }
        }
        else
        {
            if (DATA_DISTRIBUTION == ZIPFIAN)
            {
                if (Loc != 0)
                {
                    existedTID = __ldg(&log->YCSB_W[Loc]);
                }
            }
            else
            {
                existedTID = __ldg(&log->YCSB_W[Loc]);
            }
        }
        if (TID > existedTID)
        {
            atomicAdd(&CONFLICT_MARK[NO], 1);
        }
    }
    __device__ void __WRITEBACK(unsigned int NO,
                                unsigned int OP,
                                unsigned int column,
                                unsigned int Loc,
                                unsigned int query_cnt,
                                unsigned long long table_size,
                                unsigned int column_size,
                                unsigned int *set_local_set,
                                unsigned int *table)
    {
        unsigned int index = OP * YCSB_COLUMN * query_cnt + column * query_cnt + NO;
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
                                     YCSB_LOG *log,
                                     unsigned int *set_local_set,
                                     unsigned int *table)
    {
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
        const unsigned int warpId = threadIdx.x / WARP_SIZE;
        __shared__ unsigned int warpLevelSum[16];
        if (blockIdx.x == 0)
        {
            for (size_t j = 0; j < (YCSB_A_SIZE / WRITEBACK_BLOCK_SIZE == 0 ? 1 : YCSB_A_SIZE / WRITEBACK_BLOCK_SIZE); j++)
            {
                unsigned int YTD = 0;
                if (threadIdx.x + j * WRITEBACK_BLOCK_SIZE < YCSB_A_SIZE && isCommit == 0)
                {
                    YTD = (Loc == 0 ? log->YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] : 0);
                }
                YTD += __shfl_up_sync(0xffffffff, YTD, 16);
                YTD += __shfl_up_sync(0xffffffff, YTD, 8);
                YTD += __shfl_up_sync(0xffffffff, YTD, 4);
                YTD += __shfl_up_sync(0xffffffff, YTD, 2);
                YTD += __shfl_up_sync(0xffffffff, YTD, 1);
                if (laneId == 0)
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
                    YTD += log->TMP_YTD[0];
                }
                if (Loc == 0 && threadIdx.x + j * WRITEBACK_BLOCK_SIZE < YCSB_A_SIZE)
                {
                    log->YTD[threadIdx.x + j * WRITEBACK_BLOCK_SIZE] = YTD;
                    atomicMax(&log->TMP_YTD[0], YTD);
                    // printf("%d, %d, %d\n", NO, Loc, YTD);
                }
            }
        }
    }
}