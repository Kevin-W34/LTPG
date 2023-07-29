#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Query.h"
#include "Database.h"
#include "Genericfunction.h"

namespace YCSB_A_NAMESPACE
{
    __device__ void read_0(unsigned int NO,
                           unsigned int OP,
                           YCSB_A_SET *ycsb_set,
                           YCSB_A_QUERY *ycsb_query,
                           YCSB_LOG *log,
                           YCSB_SNAPSHOT *snapshot,
                           YCSB_INDEX *index)
    {
        unsigned int TID = __ldg(&ycsb_query->TID[NO]);
        unsigned int Loc = __ldg(&ycsb_query->LOC_R[YCSB_A_SIZE * OP + NO]);
        unsigned int log_loc_up = (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        ycsb_set->set_Loc[YCSB_A_SIZE * OP + NO] = Loc;
        // printf("n op_0_0 %d\n", TID);
        YCSB::__REGISTER(0, NO, OP, YCSB_A_SIZE, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void read_1(unsigned int NO,
                           unsigned int OP,
                           YCSB_A_SET *ycsb_set,
                           YCSB_A_QUERY *ycsb_query,
                           YCSB_LOG *log,
                           YCSB_SNAPSHOT *snapshot,
                           YCSB_INDEX *index)
    {
        unsigned int Loc = __ldg(&ycsb_query->LOC_R[YCSB_A_SIZE * OP + NO]);
        YCSB::__READ(NO, OP, 1, Loc, YCSB_A_SIZE, YCSB_SIZE, YCSB_COLUMN, 0, ycsb_set->set_local_set, snapshot->data);
    }
    __device__ void write_0(unsigned int NO,
                            unsigned int OP,
                            YCSB_A_SET *ycsb_set,
                            YCSB_A_QUERY *ycsb_query,
                            YCSB_LOG *log,
                            YCSB_SNAPSHOT *snapshot,
                            YCSB_INDEX *index)
    {
        unsigned int TID = __ldg(&ycsb_query->TID[NO]);
        unsigned int Loc = __ldg(&ycsb_query->LOC_W[YCSB_A_SIZE * OP + NO]);
        unsigned int log_loc_up = (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        ycsb_set->set_Loc[YCSB_A_SIZE * (YCSB_READ_SIZE + OP) + NO] = Loc;
        // printf("n op_0_0 %d\n", TID);
        YCSB::__REGISTER(0, NO, OP, YCSB_A_SIZE, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void write_1(unsigned int NO,
                            unsigned int OP,
                            YCSB_A_SET *ycsb_set,
                            YCSB_A_QUERY *ycsb_query,
                            YCSB_LOG *log,
                            YCSB_SNAPSHOT *snapshot,
                            YCSB_INDEX *index)
    {
        unsigned int Loc = __ldg(&ycsb_query->LOC_W[YCSB_A_SIZE * OP + NO]);
        if (DATA_DISTRIBUTION == ZIPFIAN)
        {
            if (Loc != 0)
            {
                YCSB::__WRITE(NO, OP, 1, Loc, YCSB_A_SIZE, YCSB_SIZE, YCSB_COLUMN, 0, ycsb_set->set_local_set, snapshot->data);
            }
            else
            {
                log->YTD[NO] = 1;
            }
        }
        else
        {
            YCSB::__WRITE(NO, OP, 1, Loc, YCSB_A_SIZE, YCSB_SIZE, YCSB_COLUMN, 0, ycsb_set->set_local_set, snapshot->data);
        }
    }
    __device__ void execute(unsigned int NO,
                            YCSB_A_SET *ycsb_set,
                            YCSB_A_QUERY *ycsb_query,
                            YCSB_LOG *log,
                            YCSB_SNAPSHOT *snapshot,
                            YCSB_INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < YCSB_A_SIZE)
        {
            // printf("execute n NO %d\n", NO);
            if ((wID % (YCSB_OP_SIZE * 2)) < (YCSB_READ_SIZE))
            {
                read_0(NO, (wID % (YCSB_OP_SIZE * 2)), ycsb_set, ycsb_query, log, snapshot, index);
            }
            else if ((wID % (YCSB_OP_SIZE * 2)) < (YCSB_READ_SIZE * 2))
            {
                read_1(NO, (wID % (YCSB_OP_SIZE * 2)) - YCSB_READ_SIZE, ycsb_set, ycsb_query, log, snapshot, index);
            }
            else if ((wID % (YCSB_OP_SIZE * 2)) < (YCSB_READ_SIZE * 2 + YCSB_WRITE_SIZE))
            {
                write_0(NO, (wID % (YCSB_OP_SIZE * 2) - (YCSB_READ_SIZE * 2)), ycsb_set, ycsb_query, log, snapshot, index);
            }
            else if ((wID % (YCSB_OP_SIZE * 2)) < (YCSB_READ_SIZE * 2 + YCSB_WRITE_SIZE * 2))
            {
                write_1(NO, (wID % (YCSB_OP_SIZE * 2) - (YCSB_READ_SIZE * 2) - YCSB_WRITE_SIZE), ycsb_set, ycsb_query, log, snapshot, index);
            }
        }
    }
    __device__ void check_r(unsigned int NO,
                            unsigned int OP,
                            YCSB_A_SET *ycsb_set,
                            YCSB_A_QUERY *ycsb_query,
                            YCSB_LOG *log,
                            YCSB_SNAPSHOT *snapshot,
                            YCSB_INDEX *index)
    {
        unsigned int TID = __ldg(&ycsb_query->TID[NO]);
        unsigned int Loc = __ldg(&ycsb_set->set_Loc[YCSB_A_SIZE * OP + NO]);
        unsigned int log_loc_up = (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        YCSB::__CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, ycsb_set->raw, log);
    }
    __device__ void check_w_0(unsigned int NO,
                              unsigned int OP,
                              YCSB_A_SET *ycsb_set,
                              YCSB_A_QUERY *ycsb_query,
                              YCSB_LOG *log,
                              YCSB_SNAPSHOT *snapshot,
                              YCSB_INDEX *index)
    {
        unsigned int TID = __ldg(&ycsb_query->TID[NO]);
        unsigned int Loc = __ldg(&ycsb_set->set_Loc[YCSB_A_SIZE * (YCSB_READ_SIZE + OP) + NO]);
        unsigned int log_loc_up = (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        YCSB::__CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, ycsb_set->war, log);
    }
    __device__ void check_w_1(unsigned int NO,
                              unsigned int OP,
                              YCSB_A_SET *ycsb_set,
                              YCSB_A_QUERY *ycsb_query,
                              YCSB_LOG *log,
                              YCSB_SNAPSHOT *snapshot,
                              YCSB_INDEX *index)
    {
        unsigned int TID = __ldg(&ycsb_query->TID[NO]);
        unsigned int Loc = __ldg(&ycsb_set->set_Loc[YCSB_A_SIZE * (YCSB_READ_SIZE + OP) + NO]);
        unsigned int log_loc_up = (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        YCSB::__CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, ycsb_set->waw, log);
    }
    __device__ void check(unsigned int NO,
                          YCSB_A_SET *ycsb_set,
                          YCSB_A_QUERY *ycsb_query,
                          YCSB_LOG *log,
                          YCSB_SNAPSHOT *snapshot,
                          YCSB_INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < YCSB_A_SIZE)
        {
            if (wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) < YCSB_READ_SIZE)
            {
                check_r(NO, wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE), ycsb_set, ycsb_query, log, snapshot, index);
            }
            else if (wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) < (YCSB_READ_SIZE + YCSB_WRITE_SIZE))
            {
                check_w_0(NO, (wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) - YCSB_READ_SIZE), ycsb_set, ycsb_query, log, snapshot, index);
            }
            else if (wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) < (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE))
            {
                check_w_1(NO, (wID % (YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) - YCSB_READ_SIZE - YCSB_WRITE_SIZE), ycsb_set, ycsb_query, log, snapshot, index);
            }
        }
    }
    __device__ void writeback_0(unsigned int NO,
                                unsigned int OP,
                                YCSB_A_SET *ycsb_set,
                                YCSB_A_QUERY *ycsb_query,
                                YCSB_LOG *log,
                                YCSB_SNAPSHOT *snapshot,
                                YCSB_INDEX *index)
    {
        unsigned int raw = __ldg(&ycsb_set->raw[NO]);
        unsigned int war = __ldg(&ycsb_set->war[NO]);
        unsigned int waw = __ldg(&ycsb_set->waw[NO]);
        if ((waw == 0) && (raw == 0 || war == 0))
        {
            atomicExch(&ycsb_set->COMMIT_AND_ABORT[NO], 1);
            atomicAdd(&ycsb_set->COMMIT, 1);
        }
    }
    __device__ void writeback_w(unsigned int NO,
                                unsigned int OP,
                                YCSB_A_SET *ycsb_set,
                                YCSB_A_QUERY *ycsb_query,
                                YCSB_LOG *log,
                                YCSB_SNAPSHOT *snapshot,
                                YCSB_INDEX *index)
    {
        unsigned int Loc = __ldg(&ycsb_set->set_Loc[YCSB_A_SIZE * (YCSB_READ_SIZE + OP) + NO]);
        if (Loc != 0)
        {
            YCSB::__WRITEBACK(NO, OP, 0, Loc, YCSB_A_SIZE, YCSB_SIZE, YCSB_COLUMN, ycsb_set->set_local_set, snapshot->data);
        }
    }
    __device__ void writeback_1(unsigned int NO,
                                unsigned int OP,
                                YCSB_A_SET *ycsb_set,
                                YCSB_A_QUERY *ycsb_query,
                                YCSB_LOG *log,
                                YCSB_SNAPSHOT *snapshot,
                                YCSB_INDEX *index)
    {
        unsigned int Loc = __ldg(&ycsb_set->set_Loc[YCSB_A_SIZE * (YCSB_READ_SIZE + OP) + NO]);
        if (Loc == 0)
        {
            unsigned int raw = __ldg(&ycsb_set->raw[NO]);
            unsigned int war = __ldg(&ycsb_set->war[NO]);
            unsigned int waw = __ldg(&ycsb_set->waw[NO]);
            unsigned int isCommit = (waw == 0) && (raw == 0 || war == 0);
            YCSB::__REDUCE_UPGRADE(NO, OP, 0, Loc, YCSB_A_SIZE, isCommit, 0, log, ycsb_set->set_local_set, snapshot->data);
        }
    }
    __device__ void write_back(unsigned int NO,
                               YCSB_A_SET *ycsb_set,
                               YCSB_A_QUERY *ycsb_query,
                               YCSB_LOG *log,
                               YCSB_SNAPSHOT *snapshot,
                               YCSB_INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < YCSB_A_SIZE)
        {
            if (wID % (1 + YCSB_WRITE_SIZE) == YCSB_WRITE_SIZE)
            {

                writeback_0(NO, 0, ycsb_set, ycsb_query, log, snapshot, index);
            }
            else
            {
                unsigned int raw = __ldg(&ycsb_set->raw[NO]);
                unsigned int war = __ldg(&ycsb_set->war[NO]);
                unsigned int waw = __ldg(&ycsb_set->waw[NO]);
                if ((waw == 0) && (raw == 0 || war == 0))
                {
                    writeback_w(NO, wID % (1 + YCSB_WRITE_SIZE), ycsb_set, ycsb_query, log, snapshot, index);
                }
            }
        }
    }
    __device__ void reduce_update(unsigned int NO,
                                  YCSB_A_SET *ycsb_set,
                                  YCSB_A_QUERY *ycsb_query,
                                  YCSB_LOG *log,
                                  YCSB_SNAPSHOT *snapshot,
                                  YCSB_INDEX *index)
    {
        if (NO < YCSB_A_SIZE)
        {
            if (blockIdx.x < YCSB_WRITE_SIZE)
            {
                // for (size_t j = 0; j < YCSB_WRITE_SIZE; j++)
                // {
                unsigned int OP = blockIdx.x;
                writeback_1(NO, OP, ycsb_set, ycsb_query, log, snapshot, index);
                // }
            }
        }
    }
}