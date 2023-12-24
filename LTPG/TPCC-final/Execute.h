#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include "Query.h"
#include "Database.h"
#include "Genericfunction.h"

namespace NEWORDER_NAMESPACE
{
    // sub-transaction execution for neworder transaction
    __device__ void op_0_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select warehouse
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int Loc = WID;
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        neworder_set->set_Loc[NEWORDER_CNT * 0 + NO] = Loc;
        // printf("n op_0_0 %d\n", TID);
        __REGISTER(0, NO, 0, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_0_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select warehouse
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int Loc = WID;
        __READ(NO, 0, 1, Loc, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, neworder_set->set_local_set, snapshot->warehouse_snapshot);
    }
    __device__ void op_1_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select district
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        neworder_set->set_Loc[NEWORDER_CNT * 1 + NO] = Loc;
        __REGISTER(0, NO, 1, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_1_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select district
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        __READ(NO, 1, 2, Loc, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, neworder_set->set_local_set, snapshot->district_snapshot);
        __READ(NO, 1, 4, Loc, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, neworder_set->set_local_set, snapshot->district_snapshot);
    }
    __device__ void op_2_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int CID = __ldg(&neworder_query->C_ID[NO]);
        unsigned int Loc = WID * 30000 + DID * 3000 + CID;
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        neworder_set->set_Loc[NEWORDER_CNT * 2 + NO] = Loc;
        __REGISTER(0, NO, 2, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_2_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int CID = __ldg(&neworder_query->C_ID[NO]);
        unsigned int Loc = WID * 30000 + DID * 3000 + CID;
        __READ(NO, 2, 4, Loc, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 2, 10, Loc, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 2, 13, Loc, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_3_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert neworder_set
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int NOID = __ldg(&neworder_query->N_O_ID[NO]);
        unsigned int Loc = WID * 30000 + NOID;
        unsigned int log_loc_up = (NEWORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        neworder_set->set_Loc[NEWORDER_CNT * 3 + NO] = Loc;
        __REGISTER(0, NO, 3, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_3_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert neworder_set
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int NOID = __ldg(&neworder_query->N_O_ID[NO]);
        unsigned int Loc = WID * 30000 + NOID;
        __WRITE(NO, 3, 0, Loc, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
        __WRITE(NO, 3, 1, Loc, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
        __WRITE(NO, 3, 2, Loc, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
    }
    __device__ void op_4_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert order
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int OID = __ldg(&neworder_query->O_ID[NO]);
        unsigned int Loc = WID * 30000 + OID;
        unsigned int log_loc_up = (ORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        neworder_set->set_Loc[NEWORDER_CNT * 4 + NO] = Loc;
        __REGISTER(0, NO, 4, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_4_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert order
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int OID = __ldg(&neworder_query->O_ID[NO]);
        unsigned int Loc = WID * 30000 + OID;
        __WRITE(NO, 4, 0, Loc, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
        __WRITE(NO, 4, 1, Loc, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
        __WRITE(NO, 4, 2, Loc, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
    }
    __device__ void op_5_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select item
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 10;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int TID = __ldg(&neworder_query->TID[NO]);
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = IID;
            unsigned int log_loc_up = (ITEM_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
            neworder_set->set_Loc[NEWORDER_CNT * (5 + ID) + NO] = Loc;
            __REGISTER(0, NO, 5 + ID, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        }
    }
    __device__ void op_5_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select item
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 25;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = IID;
            __READ(NO, 5 + ID, 2, Loc, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
            __READ(NO, 5 + ID, 3, Loc, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
            __READ(NO, 5 + ID, 4, Loc, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
        }
    }
    __device__ void op_6_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 40;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int TID = __ldg(&neworder_query->TID[NO]);
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int SWID = __ldg(&neworder_query->OL_SUPPLY_W_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = SWID * 100000 + IID;
            unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            neworder_set->set_Loc[NEWORDER_CNT * (20 + ID) + NO] = Loc;
            __REGISTER(0, NO, 20 + ID, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        }
    }
    __device__ void op_6_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 55;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int SWID = __ldg(&neworder_query->OL_SUPPLY_W_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = SWID * 100000 + IID;
            __WRITE(NO, 20 + ID, 2, Loc, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
            __WRITE(NO, 20 + ID, 3, Loc, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
            __WRITE(NO, 20 + ID, 4, Loc, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
        }
    }
    __device__ void op_7_0(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert orderline

        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 70;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int TID = __ldg(&neworder_query->TID[NO]);
            unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
            unsigned int OOLID = __ldg(&neworder_query->O_OL_ID[NO]);
            unsigned int Loc = WID * 450000 + OOLID + ID;
            unsigned int log_loc_up = (ORDERLINE_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO] = Loc;
            __REGISTER(0, NO, 35 + ID, NEWORDER_CNT, TID, Loc, log_loc_up, log_loc_down, log);
            __WRITE(NO, 35 + ID, 0, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void op_7_1(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 85;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
            unsigned int OOLID = __ldg(&neworder_query->O_OL_ID[NO]);
            unsigned int Loc = WID * 450000 + OOLID + ID;
            __WRITE(NO, 35 + ID, 1, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 2, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 3, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void op_7_2(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 100;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
            unsigned int OOLID = __ldg(&neworder_query->O_OL_ID[NO]);
            unsigned int Loc = WID * 450000 + OOLID + ID;
            __WRITE(NO, 35 + ID, 4, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 5, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 6, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void op_7_3(unsigned int NO,
                           NEWORDER_SET *neworder_set,
                           NEWORDER_QUERY *neworder_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 115;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
            unsigned int OOLID = __ldg(&neworder_query->O_OL_ID[NO]);
            unsigned int Loc = WID * 450000 + OOLID + ID;
            __WRITE(NO, 35 + ID, 7, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 8, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITE(NO, 35 + ID, 9, Loc, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }

    __device__ void execute(unsigned int NO,
                            NEWORDER_SET *neworder_set,
                            NEWORDER_QUERY *neworder_query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    { // execute transactions with adaptive warps division
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < NEWORDER_CNT)
        {
            if (wID % EXECUTE_WARP == 0)
            {
                op_0_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 1)
            {
                op_0_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 2)
            {
                op_1_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 3)
            {
                op_1_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 4)
            {
                op_2_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 5)
            {
                op_2_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 6)
            {
                op_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 7)
            {
                op_3_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 8)
            {
                op_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 9)
            {
                op_4_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 25)
            {
                op_5_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 40)
            {
                op_5_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 55)
            {
                op_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 70)
            {
                op_6_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 85)
            {
                op_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 100)
            {
                op_7_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 115)
            {
                op_7_2(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 130)
            {
                op_7_3(NO, neworder_set, neworder_query, log, snapshot, index);
            }
        }
    }

    // check conflict of transactions
    __device__ void check_0_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select warehouse
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 0 + NO]);
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->raw, log);
    }
    __device__ void check_1_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select district
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 1 + NO]);
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->raw, log);
    }
    __device__ void check_2_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select customer
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 2 + NO]);
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->raw, log);
    }
    __device__ void check_3_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert neworder_set
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 3 + NO]);
        unsigned int log_loc_up = (NEWORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->war, log);
    }
    __device__ void check_3_1(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert neworder_set
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 3 + NO]);
        unsigned int log_loc_up = (NEWORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->waw, log);
    }
    __device__ void check_4_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert order
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 4 + NO]);
        unsigned int log_loc_up = (ORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->war, log);
    }
    __device__ void check_4_1(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert order
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 4 + NO]);
        unsigned int log_loc_up = (ORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->waw, log);
    }
    __device__ void check_5_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select item
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 7;
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (5 + ID) + NO]);
            unsigned int log_loc_up = (ITEM_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->raw, log);
        }
    }
    __device__ void check_6_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 22;
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (20 + ID) + NO]);
            unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->war, log);
        }
    }
    __device__ void check_6_1(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 37;
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (20 + ID) + NO]);
            unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->waw, log);
        }
    }
    __device__ void check_7_0(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 52;
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO]);
            unsigned int log_loc_up = (ORDERLINE_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->war, log);
        }
    }
    __device__ void check_7_1(unsigned int NO,
                              NEWORDER_SET *neworder_set,
                              NEWORDER_QUERY *neworder_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert orderline

        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 67;
        unsigned int TID = __ldg(&neworder_query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO]);
            unsigned int log_loc_up = (ORDERLINE_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, neworder_set->waw, log);
        }
    }

    __device__ void check(unsigned int NO,
                          NEWORDER_SET *neworder_set,
                          NEWORDER_QUERY *neworder_query,
                          LOG *log,
                          SNAPSHOT *snapshot,
                          INDEX *index)
    { // check conflict
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < NEWORDER_CNT)
        {
            if (wID % CHECK_WARP == 0)
            {
                check_0_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 1)
            {
                check_1_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 3)
            {
                check_2_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 3)
            {
                check_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 4)
            {
                check_3_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 5)
            {
                check_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 6)
            {
                check_4_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 22)
            {
                check_5_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 37)
            {
                check_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 52)
            {
                check_6_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 67)
            {
                check_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 82)
            {
                check_7_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
        }
    }
    __device__ void writeback_0(unsigned int NO,
                                NEWORDER_SET *neworder_set,
                                NEWORDER_QUERY *neworder_query,
                                LOG *log,
                                SNAPSHOT *snapshot,
                                INDEX *index)
    { // insert neworder_set 3
        unsigned int raw = __ldg(&neworder_set->raw[NO]);
        unsigned int war = __ldg(&neworder_set->war[NO]);
        unsigned int waw = __ldg(&neworder_set->waw[NO]);
        // printf("n TID %d, waw %d, war %d, raw %d\n", __ldg(&neworder_query->TID[NO]), waw, war, raw);
        if ((waw == 0) && (raw == 0 || war == 0))
        {
            atomicExch(&neworder_set->COMMIT_AND_ABORT[NO], 1);
            atomicAdd(&neworder_set->COMMIT, 1);
        }
    }
    __device__ void writeback_3_0(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert neworder_set 3
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 3 + NO]);
        __WRITEBACK(NO, 3, 0, Loc, NEWORDER_CNT, NEWORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
        __WRITEBACK(NO, 3, 1, Loc, NEWORDER_CNT, NEWORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
        __WRITEBACK(NO, 3, 2, Loc, NEWORDER_CNT, NEWORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
    }
    __device__ void writeback_4_0(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert order 8
        unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * 4 + NO]);
        __WRITEBACK(NO, 4, 0, Loc, NEWORDER_CNT, ORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->order_snapshot);
        __WRITEBACK(NO, 4, 1, Loc, NEWORDER_CNT, ORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->order_snapshot);
        __WRITEBACK(NO, 4, 2, Loc, NEWORDER_CNT, ORDER_SIZE, 0, 0, neworder_set->set_local_set, snapshot->order_snapshot);
    }
    __device__ void writeback_6_0(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // update stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % WRITEBACK_WARP - 3;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (20 + ID) + NO]);
            __WRITEBACK(NO, 20 + ID, 2, Loc, NEWORDER_CNT, STOCK_SIZE, 0, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
            __WRITEBACK(NO, 20 + ID, 3, Loc, NEWORDER_CNT, STOCK_SIZE, 0, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
            __WRITEBACK(NO, 20 + ID, 4, Loc, NEWORDER_CNT, STOCK_SIZE, 0, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
        }
    }
    __device__ void writeback_7_0(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % WRITEBACK_WARP - 18;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO]);
            __WRITEBACK(NO, 35 + ID, 0, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 1, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 2, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void writeback_7_1(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % WRITEBACK_WARP - 33;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO]);
            __WRITEBACK(NO, 35 + ID, 3, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 4, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 5, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void writeback_7_2(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert orderline
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % WRITEBACK_WARP - 48;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&neworder_set->set_Loc[NEWORDER_CNT * (35 + ID) + NO]);
            __WRITEBACK(NO, 35 + ID, 6, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 7, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 8, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
            __WRITEBACK(NO, 35 + ID, 9, Loc, NEWORDER_CNT, ORDERLINE_SIZE, 0, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
        }
    }
    __device__ void write_back(unsigned int NO,
                               NEWORDER_SET *neworder_set,
                               NEWORDER_QUERY *neworder_query,
                               LOG *log,
                               SNAPSHOT *snapshot,
                               INDEX *index)
    {
        // write_back_op<<<NEWORDER_WB_TYPE_CNT, 1, 0,cudaStreamPerThread  >>>(NO, neworder_set, log, snapshot, index);
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < NEWORDER_CNT)
        {
            if (wID % WRITEBACK_WARP == 0)
            {
                writeback_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else
            {
                unsigned int raw = __ldg(&neworder_set->raw[NO]);
                unsigned int war = __ldg(&neworder_set->war[NO]);
                unsigned int waw = __ldg(&neworder_set->waw[NO]);
                if ((waw == 0) && (raw == 0 || war == 0))
                {
                    if (wID % WRITEBACK_WARP == 1)
                    {
                        writeback_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    if (wID % WRITEBACK_WARP == 2)
                    {
                        writeback_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID % WRITEBACK_WARP < 18)
                    {
                        writeback_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID % WRITEBACK_WARP < 33)
                    {
                        writeback_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID % WRITEBACK_WARP < 48)
                    {
                        writeback_7_1(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID % WRITEBACK_WARP < 63)
                    {
                        writeback_7_2(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                }
            }
        }
    }
} // namespace NEWORDERQUERY

namespace PAYMENT_NAMESPACE
{
    __device__ void get_c_id(unsigned int NO,
                             PAYMENT_SET *payment_set,
                             PAYMENT_QUERY *payment_query,
                             LOG *log,
                             SNAPSHOT *snapshot,
                             INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (NO < PAYMENT_CNT)
        {
            unsigned int isName = __ldg(&payment_query->isName[NO]);
            if (isName)
            {
                unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
                unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
                unsigned int C_LAST = __ldg(&payment_query->C_LAST[NO]);
                unsigned int C_ID = __FIND_CID_INDEX(C_LAST, CWID * 30000 + CDID * 3000, index->customer_name_index, 3000, snapshot->customer_snapshot);
                if (thID % WARP_SIZE == 0)
                {
                    payment_query->C_ID[NO] = C_ID;
                }
            }
        }
    }
    __device__ void op_0_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select warehouse
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int Loc = WID;
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 0 + NO] = Loc;
        __REGISTER(0, NO, 0, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_0_1(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select warehouse
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int Loc = WID;
        __READ(NO, 0, 3, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
        __READ(NO, 0, 4, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
        __READ(NO, 0, 5, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
    }
    __device__ void op_0_2(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select warehouse
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int Loc = WID;
        __READ(NO, 0, 6, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
        __READ(NO, 0, 7, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
        __READ(NO, 0, 8, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
    }
    __device__ void op_1_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update warehouse
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int Loc = WID;
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 1 + NO] = Loc;
        __REGISTER(1, NO, 1, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        // payment_set->set_local_set[1 * 32 * PAYMENT_CNT + 2 * PAYMENT_CNT + NO] = __ldg(&payment_query->H_AMOUNT[NO]);
        // __WRITE(NO, 1, 2, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 2, payment_set->set_local_set, snapshot->warehouse_snapshot);
        log->W_YTD[NO] = payment_query->H_AMOUNT[NO];
        // printf("%d\n", log->W_YTD[NO]);
        // log->Loc_W_YTD[NO] = Loc;
    }
    __device__ void op_2_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select district
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 2 + NO] = Loc;
        __REGISTER(0, NO, 2, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_2_1(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select district
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        __READ(NO, 2, 5, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
        __READ(NO, 2, 6, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
        __READ(NO, 2, 7, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
    }
    __device__ void op_2_2(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select district
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        __READ(NO, 2, 8, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
        __READ(NO, 2, 9, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
        __READ(NO, 2, 10, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
    }
    __device__ void op_3_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update district
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 3 + NO] = Loc;
        __REGISTER(1, NO, 3, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        // payment_set->set_local_set[3 * 32 * PAYMENT_CNT + 3 * PAYMENT_CNT + NO] = __ldg(&payment_query->H_AMOUNT[NO]);
        // __WRITE(NO, 3, 3, Loc, PAYMENT_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
        log->D_YTD[NO] = payment_query->H_AMOUNT[NO];
        // log->Loc_D_YTD[NO] = Loc;
    }
    __device__ void op_4_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 4 + NO] = Loc;
        __REGISTER(0, NO, 4, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_4_1(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        __READ(NO, 4, 3, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot); // 5
        __READ(NO, 4, 4, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 11, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_4_2(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        __READ(NO, 4, 12, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 13, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 14, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_4_3(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        __READ(NO, 4, 15, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 16, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 17, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_4_4(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        __READ(NO, 4, 18, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 19, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __READ(NO, 4, 20, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_5_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update customer
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 5 + NO] = Loc;
        __REGISTER(0, NO, 5, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_5_1(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // update customer
        unsigned int CWID = __ldg(&payment_query->C_W_ID[NO]);
        unsigned int CDID = __ldg(&payment_query->C_D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = CWID * 30000 + CDID * 3000 + CID;
        __WRITE(NO, 5, 5, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __WRITE(NO, 5, 9, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_6_0(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert history
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int HID = __ldg(&payment_query->H_ID[NO]);
        unsigned int Loc = WID * 30000 + HID;
        unsigned int log_loc_up = (HISTORY_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        payment_set->set_Loc[PAYMENT_CNT * 6 + NO] = Loc;
        __REGISTER(0, NO, 6, PAYMENT_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        __WRITE(NO, 6, 0, Loc, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITE(NO, 6, 1, Loc, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->history_snapshot);
    }
    __device__ void op_6_1(unsigned int NO,
                           PAYMENT_SET *payment_set,
                           PAYMENT_QUERY *payment_query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // insert history
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int HID = __ldg(&payment_query->H_ID[NO]);
        unsigned int Loc = WID * 30000 + HID;
        __WRITE(NO, 6, 2, Loc, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITE(NO, 6, 3, Loc, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITE(NO, 6, 4, Loc, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->history_snapshot);
    }
    __device__ void execute(unsigned int NO,
                            PAYMENT_SET *payment_set,
                            PAYMENT_QUERY *payment_query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < PAYMENT_CNT)
        {
            // printf("execute p NO %d\n", NO);
            if (wID % EXECUTE_WARP == 130)
            {
                op_0_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 131)
            {
                op_0_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 132)
            {
                op_0_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 133)
            {
                op_1_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 134)
            {
                op_2_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 135)
            {
                op_2_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 136)
            {
                op_2_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 137)
            {
                op_3_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 138)
            {
                op_4_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 139)
            {
                op_4_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 140)
            {
                op_4_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 141)
            {
                op_4_3(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 142)
            {
                op_4_4(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 143)
            {
                op_5_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 144)
            {
                op_5_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 145)
            {
                op_6_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 146)
            {
                op_6_1(NO, payment_set, payment_query, log, snapshot, index);
            }
        }
    }
    __device__ void check_0_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select warehouse
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 0 + NO]);
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->raw, log);
    }
    __device__ void check_1_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update warehouse
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 1 + NO]);
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        // __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
        __CHECK(1, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
    }
    __device__ void check_1_1(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update warehouse
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 1 + NO]);
        unsigned int log_loc_up = (WAREHOUSE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        // __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
        __CHECK(1, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
    }
    __device__ void check_2_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select district
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 2 + NO]);
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->raw, log);
    }
    __device__ void check_3_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update district
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 3 + NO]);
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        // __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
        __CHECK(1, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
    }
    __device__ void check_3_1(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update district
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 3 + NO]);
        unsigned int log_loc_up = (DISTRICT_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        // __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
        __CHECK(1, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
    }
    __device__ void check_4_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // select customer
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 4 + NO]);
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->raw, log);
    }
    __device__ void check_5_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update customer
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 5 + NO]);
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
    }
    __device__ void check_5_1(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // update customer
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 5 + NO]);
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
    }
    __device__ void check_6_0(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert history
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 6 + NO]);
        unsigned int log_loc_up = (HISTORY_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->war, log);
    }
    __device__ void check_6_1(unsigned int NO,
                              PAYMENT_SET *payment_set,
                              PAYMENT_QUERY *payment_query,
                              LOG *log,
                              SNAPSHOT *snapshot,
                              INDEX *index)
    { // insert history
        unsigned int TID = __ldg(&payment_query->TID[NO]);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 6 + NO]);
        unsigned int log_loc_up = (HISTORY_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, payment_set->waw, log);
    }

    __device__ void check(unsigned int NO,
                          PAYMENT_SET *payment_set,
                          PAYMENT_QUERY *payment_query,
                          LOG *log,
                          SNAPSHOT *snapshot,
                          INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < PAYMENT_CNT)
        {
            if (wID % CHECK_WARP == 82)
            {
                check_0_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 83)
            {
                check_1_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 84)
            {
                check_1_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 85)
            {
                check_2_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 86)
            {
                check_3_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 87)
            {
                check_3_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 88)
            {
                check_4_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 89)
            {
                check_5_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 90)
            {
                check_5_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 91)
            {
                check_6_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP == 92)
            {
                check_6_1(NO, payment_set, payment_query, log, snapshot, index);
            }
        }
    }
    __device__ void writeback_0(unsigned int NO,
                                PAYMENT_SET *payment_set,
                                PAYMENT_QUERY *payment_query,
                                LOG *log,
                                SNAPSHOT *snapshot,
                                INDEX *index)
    { // insert neworder_set 3
        unsigned int raw = __ldg(&payment_set->raw[NO]);
        unsigned int war = __ldg(&payment_set->war[NO]);
        unsigned int waw = __ldg(&payment_set->waw[NO]);
        // printf("p TID %d, waw %d, war %d, raw %d\n", __ldg(&payment_query->TID[NO]), waw, war, raw);
        if ((waw == 0) && (raw == 0 || war == 0))
        {
            atomicExch(&payment_set->COMMIT_AND_ABORT[NO], 1);
            atomicAdd(&payment_set->COMMIT, 1);
        }
    }
    __device__ void writeback_1_0(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // update warehouse
        unsigned int raw = __ldg(&payment_set->raw[NO]);
        unsigned int war = __ldg(&payment_set->war[NO]);
        unsigned int waw = __ldg(&payment_set->waw[NO]);
        unsigned int isCommit = (waw == 0) && (raw == 0 || war == 0);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 1 + NO]);
        // __WRITEBACK(NO, 1, 2, Loc, PAYMENT_CNT, WAREHOUSE_SIZE, 0, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
        __REDUCE_UPGRADE(NO, 1, 2, Loc, PAYMENT_CNT, isCommit, WAREHOUSE_ID, log, payment_set->set_local_set, snapshot->warehouse_snapshot);
    }
    __device__ void writeback_3_0(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // update district
        unsigned int raw = __ldg(&payment_set->raw[NO]);
        unsigned int war = __ldg(&payment_set->war[NO]);
        unsigned int waw = __ldg(&payment_set->waw[NO]);
        unsigned int isCommit = (waw == 0) && (raw == 0 || war == 0);
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 3 + NO]);
        // __WRITEBACK(NO, 3, 3, Loc, PAYMENT_CNT, DISTRICT_SIZE, 0, 0, payment_set->set_local_set, snapshot->district_snapshot);
        __REDUCE_UPGRADE(NO, 3, 3, Loc, PAYMENT_CNT, isCommit, DISTRICT_ID, log, payment_set->set_local_set, snapshot->district_snapshot);
    }
    __device__ void writeback_5_0(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // update customer
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 5 + NO]);
        __WRITEBACK(NO, 5, 5, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __WRITEBACK(NO, 5, 9, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        // __REDUCE_UPGRADE(NO, 5, 5, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_ID, log, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void writeback_5_1(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // update customer
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 5 + NO]);
        __WRITEBACK(NO, 5, 5, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        __WRITEBACK(NO, 5, 9, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        // __REDUCE_UPGRADE(NO, 5, 9, Loc, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_ID, log, payment_set->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void writeback_6_0(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert history
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 6 + NO]);
        __WRITEBACK(NO, 6, 0, Loc, PAYMENT_CNT, HISTORY_SIZE, 0, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITEBACK(NO, 6, 1, Loc, PAYMENT_CNT, HISTORY_SIZE, 0, 0, payment_set->set_local_set, snapshot->history_snapshot);
    }
    __device__ void writeback_6_1(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    { // insert history
        unsigned int Loc = __ldg(&payment_set->set_Loc[PAYMENT_CNT * 6 + NO]);
        __WRITEBACK(NO, 6, 2, Loc, PAYMENT_CNT, HISTORY_SIZE, 0, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITEBACK(NO, 6, 3, Loc, PAYMENT_CNT, HISTORY_SIZE, 0, 0, payment_set->set_local_set, snapshot->history_snapshot);
        __WRITEBACK(NO, 6, 4, Loc, PAYMENT_CNT, HISTORY_SIZE, 0, 0, payment_set->set_local_set, snapshot->history_snapshot);
    }
    __device__ void write_back(unsigned int NO,
                               PAYMENT_SET *payment_set,
                               PAYMENT_QUERY *payment_query,
                               LOG *log,
                               SNAPSHOT *snapshot,
                               INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < PAYMENT_CNT)
        {
            if (wID % WRITEBACK_WARP == 63)
            {
                writeback_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else
            {
                unsigned int raw = __ldg(&payment_set->raw[NO]);
                unsigned int war = __ldg(&payment_set->war[NO]);
                unsigned int waw = __ldg(&payment_set->waw[NO]);
                if ((waw == 0) && (raw == 0 || war == 0))
                {
                    if (wID % WRITEBACK_WARP == 64)
                    {
                        writeback_6_0(NO, payment_set, payment_query, log, snapshot, index);
                    }
                    else if (wID % WRITEBACK_WARP == 65)
                    {
                        writeback_6_1(NO, payment_set, payment_query, log, snapshot, index);
                    }
                    // else if (wID % WRITEBACK_WARP == 66)
                    // {
                    //     // writeback_1_0(NO, payment_set, payment_query, log, snapshot, index);
                    // }
                    // else if (wID % WRITEBACK_WARP == 67)
                    // {
                    //     // writeback_3_0(NO, payment_set, payment_query, log, snapshot, index);
                    // }
                    else if (wID % WRITEBACK_WARP == 66)
                    {
                        writeback_5_0(NO, payment_set, payment_query, log, snapshot, index);
                    }
                }
            }
        }
    }
    __device__ void reduce_update(unsigned int NO,
                                  PAYMENT_SET *payment_set,
                                  PAYMENT_QUERY *payment_query,
                                  LOG *log,
                                  SNAPSHOT *snapshot,
                                  INDEX *index)
    {
        if (NO < PAYMENT_CNT)
        {
            if (blockIdx.x < WAREHOUSE_SIZE)
            {
                writeback_1_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            if (blockIdx.x < DISTRICT_SIZE)
            {
                writeback_3_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            // if (blockIdx.x < CUSTOMER_SIZE)
            // {
            //     writeback_5_0(NO, payment_set, payment_query, log, snapshot, index);
            // }
            // if (blockIdx.x < CUSTOMER_SIZE)
            // {
            //     writeback_5_1(NO, payment_set, payment_query, log, snapshot, index);
            // }
        }
    }
} // namespace PAYMENTQUERY

namespace ORDERSTATUSQUERY_NAMESPACE
{
    __device__ void op_0_0(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int WID = __ldg(&query[NO].query->W_ID);
        unsigned int DID = __ldg(&query[NO].query->D_ID);
        unsigned int CID = __ldg(&query[NO].query->C_ID);
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = WID * 30000 + DID * 3000 + CID;
        unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        query->set_Loc[ORDERSTATUS_CNT * 0 + NO] = Loc;
        __REGISTER(0, NO, 0, ORDERSTATUS_OP_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_0_1(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select customer
        unsigned int WID = __ldg(&query[NO].query->W_ID);
        unsigned int DID = __ldg(&query[NO].query->D_ID);
        unsigned int CID = __ldg(&query[NO].query->C_ID);
        unsigned int Loc = WID * 30000 + DID * 3000 + CID;
        __READ(NO, 0, 2, Loc, ORDERSTATUS_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, query->set_local_set, snapshot->customer_snapshot);
    }
    __device__ void op_1_0(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select order
        unsigned int WID = __ldg(&query->query[NO].W_ID);
        unsigned int OID = __ldg(&query->query[NO].O_ID);
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = WID * 30000 + OID;
        unsigned int log_loc_up = (ORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __REGISTER(0, NO, 1, ORDERSTATUS_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_1_1(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select order
        unsigned int WID = __ldg(&query->query[NO].W_ID);
        unsigned int OID = __ldg(&query->query[NO].O_ID);
        unsigned int Loc = WID * 30000 + OID;
        __READ(NO, 1, 0, Loc, ORDERSTATUS_CNT, ORDER_SIZE, ORDER_COLUMN, 0, query->set_local_set, snapshot->order_snapshot);
    }
    __device__ void op_2_0(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select orderline
        unsigned int WID = __ldg(&query->query[NO].W_ID);
        unsigned int OLID = __ldg(&query->query[NO].OL_ID);
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = WID * 450000 + OLID;
        unsigned int log_loc_up = (ORDERLINE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
        __REGISTER(0, NO, 2, ORDERSTATUS_CNT, TID, Loc, log_loc_up, log_loc_down, log);
    }
    __device__ void op_2_1(unsigned int NO,
                           ORDERSTATUS_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select orderline
        unsigned int WID = __ldg(&query->query[NO].W_ID);
        unsigned int OLID = __ldg(&query->query[NO].OL_ID);
        unsigned int Loc = WID * 450000 + OLID;
        __READ(NO, 2, 0, Loc, ORDERSTATUS_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, query->set_local_set, snapshot->orderline_snapshot);
    }
    __device__ void execute(unsigned int NO,
                            ORDERSTATUS_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < ORDERSTATUS_CNT)
        {
            if (wID % EXECUTE_WARP < 1)
            {
                op_0_0(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 2)
            {
                op_0_1(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 1)
            {
                op_1_0(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 2)
            {
                op_1_1(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP < 1)
            {
                op_2_0(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 2)
            {
                op_2_1(NO, query, log, snapshot, index);
            }
        }
    }
    __device__ void check_0(unsigned int NO,
                            ORDERSTATUS_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    { // select customer
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = __ldg(&query->set_Loc[ORDERSTATUS_CNT * 0 + NO]);
        unsigned int log_loc_up = (CUSTOMER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, query->raw, log);
    }
    __device__ void check_1(unsigned int NO,
                            ORDERSTATUS_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    { // select order
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = __ldg(&query->set_Loc[ORDERSTATUS_CNT * 1 + NO]);
        unsigned int log_loc_up = (ORDER_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(1, NO, Loc, TID, log_loc_up, log_loc_down, query->raw, log);
    }
    __device__ void check_2(unsigned int NO,
                            ORDERSTATUS_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    { // select orderline
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int Loc = __ldg(&query->set_Loc[ORDERSTATUS_CNT * 2 + NO]);
        unsigned int log_loc_up = (ORDERLINE_ID << 24) + (Loc >> 8);
        unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
        __CHECK(2, NO, Loc, TID, log_loc_up, log_loc_down, query->raw, log);
    }
    __device__ void check(unsigned int NO,
                          ORDERSTATUS_SET *query,
                          LOG *log,
                          SNAPSHOT *snapshot,
                          INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < STOCK_CNT)
        {
            if (wID % CHECK_WARP < 1)
            {
                check_0(NO, query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 1)
            {
                check_0(NO, query, log, snapshot, index);
            }
            else if (wID % CHECK_WARP < 1)
            {
                check_0(NO, query, log, snapshot, index);
            }
        }
    }
} // namespace ORDERSTATUSQUERY

namespace DELIVERYQUERY_NAMESPACE
{
    void execute(unsigned int NO,
                 DELIVERY_SET *query,
                 LOG *log,
                 SNAPSHOT *snapshot,
                 INDEX *index)
    {
    }
} // namespace DELIVERYQUERY

namespace STOCKLEVELQUERY_NAMESPACE
{
    __device__ void op_0_0(unsigned int NO,
                           STOCK_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 55;
        unsigned int O_OL_CNT = __ldg(&query[NO].query->query_cnt);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = query[NO].query->W_ID;
            unsigned int O_OL_ID = index->orderline_index[WID];
            unsigned int S_I_ID = snapshot->orderline_snapshot[WID * 450000 + O_OL_ID];
            unsigned int TID = __ldg(&query->TID[NO]);
            unsigned int Loc = WID * 100000 + S_I_ID;
            unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + READ_TYPE;
            query->set_Loc[STOCK_CNT * ID + NO] = Loc;
            __REGISTER(0, NO, ID, STOCK_OP_CNT, TID, Loc, log_loc_up, log_loc_down, log);
        }
    }
    __device__ void op_0_1(unsigned int NO,
                           STOCK_SET *query,
                           LOG *log,
                           SNAPSHOT *snapshot,
                           INDEX *index)
    { // select stock
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % EXECUTE_WARP - 55;
        unsigned int O_OL_CNT = __ldg(&query[NO].query->query_cnt);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = query[NO].query->W_ID;
            unsigned int O_OL_ID = index->orderline_index[WID];
            unsigned int S_I_ID = snapshot->orderline_snapshot[WID * 450000 + O_OL_ID];
            unsigned int Loc = WID * 100000 + S_I_ID;
            __READ(NO, ID, 2, Loc, STOCK_CNT, STOCK_SIZE, STOCK_COLUMN, 0, query->set_local_set, snapshot->stock_snapshot);
        }
    }
    __device__ void execute(unsigned int NO,
                            STOCK_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < STOCK_CNT)
        {
            if (wID % EXECUTE_WARP < 1)
            {
                op_0_0(NO, query, log, snapshot, index);
            }
            else if (wID % EXECUTE_WARP == 2)
            {
                op_0_1(NO, query, log, snapshot, index);
            }
        }
    }
    __device__ void check_0(unsigned int NO,
                            STOCK_SET *query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        unsigned int ID = wID % CHECK_WARP - 67;
        unsigned int TID = __ldg(&query->TID[NO]);
        unsigned int O_OL_CNT = __ldg(&query[NO].query->query_cnt);
        if (ID < O_OL_CNT)
        {
            unsigned int Loc = __ldg(&query->set_Loc[STOCK_CNT * ID + NO]);
            unsigned int log_loc_up = (STOCK_ID << 24) + (Loc >> 8);
            unsigned int log_loc_down = (Loc << 24) + WRITE_TYPE;
            __CHECK(0, NO, Loc, TID, log_loc_up, log_loc_down, query->raw, log);
        }
    }
    __device__ void check(unsigned int NO,
                          STOCK_SET *query,
                          LOG *log,
                          SNAPSHOT *snapshot,
                          INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < STOCK_CNT)
        {
            if (wID % CHECK_WARP < 1)
            {
                check_0(NO, query, log, snapshot, index);
            }
        }
    }
} // namespace STOCKLEVELQUERY