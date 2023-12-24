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
        wID = wID % EXECUTE_WARP;
        if (NO < NEWORDER_CNT)
        {
            if (wID == 0)
            {
                op_0_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 1)
            {
                op_0_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 2)
            {
                op_1_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 3)
            {
                op_1_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 4)
            {
                op_2_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 5)
            {
                op_2_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 6)
            {
                op_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 7)
            {
                op_3_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 8)
            {
                op_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 9)
            {
                op_4_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 25)
            {
                op_5_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 40)
            {
                op_5_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 55)
            {
                op_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 70)
            {
                op_6_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 85)
            {
                op_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 100)
            {
                op_7_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 115)
            {
                op_7_2(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 130)
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
    { // check 
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        wID = wID % CHECK_WARP;
        if (NO < NEWORDER_CNT)
        {
            if (wID == 0)
            {
                check_0_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 1)
            {
                check_1_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 3)
            {
                check_2_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 3)
            {
                check_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 4)
            {
                check_3_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 5)
            {
                check_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID == 6)
            {
                check_4_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 22)
            {
                check_5_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 37)
            {
                check_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 52)
            {
                check_6_1(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 67)
            {
                check_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
            }
            else if (wID < 82)
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
            // atomicExch(&neworder_set->COMMIT_AND_ABORT[NO], 1);
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
        wID = wID % WRITEBACK_WARP;
        if (NO < NEWORDER_CNT)
        {
            if (wID == 0)
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
                    if (wID == 1)
                    {
                        writeback_3_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    if (wID == 2)
                    {
                        writeback_4_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID < 18)
                    {
                        writeback_6_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID < 33)
                    {
                        writeback_7_0(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID < 48)
                    {
                        writeback_7_1(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                    else if (wID < 63)
                    {
                        writeback_7_2(NO, neworder_set, neworder_query, log, snapshot, index);
                    }
                }
            }
        }
    }
} // namespace NEWORDERQUERY

