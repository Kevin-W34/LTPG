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
    { // insert neworder
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
    { // insert neworder
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
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        if (NO < NEWORDER_CNT)
        {
            // printf("execute n NO %d\n", NO);
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
    {
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
        // __WRITEBACK(NO, 5, 5, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
        // __WRITEBACK(NO, 5, 9, Loc, PAYMENT_CNT, CUSTOMER_SIZE, 0, 0, payment_set->set_local_set, snapshot->customer_snapshot);
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

namespace GACCO_NEWORDER_NAMESPACE
{
    __device__ void get_warehouse_access(unsigned int NO,
                                         NEWORDER_QUERY *neworder_query,
                                         WAREHOUSE_ACCESS *warehouse_access,
                                         WAREHOUSE_AUXILIARY *warehouse_auxiliary)
    {
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int Loc = WID;
        unsigned int TID = NO;
        warehouse_access->W_ID[NO] = Loc;
        warehouse_access->TID[NO] = TID;
        atomicAdd(&warehouse_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_district_access(unsigned int NO,
                                        NEWORDER_QUERY *neworder_query,
                                        DISTRICT_ACCESS *district_access,
                                        DISTRICT_AUXILIARY *district_auxiliary)
    {
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        unsigned int TID = NO;
        district_access->D_ID[NO] = Loc;
        district_access->TID[NO] = TID;
        atomicAdd(&district_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_customer_access(unsigned int NO,
                                        NEWORDER_QUERY *neworder_query,
                                        CUSTOMER_ACCESS *customer_access,
                                        CUSTOMER_AUXILIARY *customer_auxiliary)
    {
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int DID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int CID = __ldg(&neworder_query->C_ID[NO]);
        unsigned int Loc = (WID * 10 + DID) * 3000 + CID;
        unsigned int TID = NO;
        customer_access->C_ID[NO] = Loc;
        customer_access->TID[NO] = TID;
        atomicAdd(&customer_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_neworder_access(unsigned int NO,
                                        NEWORDER_QUERY *neworder_query,
                                        NEWORDER_ACCESS *neworder_access,
                                        NEWORDER_AUXILIARY *neworder_auxiliary)
    {
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int NOID = __ldg(&neworder_query->D_ID[NO]);
        unsigned int Loc = WID * 30000 + NOID;
        unsigned int TID = NO;
        neworder_access->N_ID[NO] = Loc;
        neworder_access->TID[NO] = TID;
        atomicAdd(&neworder_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_order_access(unsigned int NO,
                                     NEWORDER_QUERY *neworder_query,
                                     ORDER_ACCESS *order_access,
                                     ORDER_AUXILIARY *order_auxiliary)
    {
        unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
        unsigned int OID = __ldg(&neworder_query->O_ID[NO]);
        unsigned int Loc = WID * 30000 + OID;
        unsigned int TID = NO;
        order_access->O_ID[NO] = Loc;
        order_access->TID[NO] = TID;
        atomicAdd(&order_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_item_access(unsigned int NO,
                                    NEWORDER_QUERY *neworder_query,
                                    ITEM_ACCESS *item_access,
                                    ITEM_AUXILIARY *item_auxiliary)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // unsigned int warpID = thID >> 5;
        unsigned int ID = 0;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = IID;
            unsigned int TID = NO;
            item_access->I_ID[NEWORDER_CNT * (5 + ID) + NO] = Loc;
            item_access->TID[NO] = TID;
            atomicAdd(&item_auxiliary->HIST[Loc], 1);
        }
    }
    __device__ void get_stock_access(unsigned int NO,
                                     NEWORDER_QUERY *neworder_query,
                                     STOCK_ACCESS *stock_access,
                                     STOCK_AUXILIARY *stock_auxiliary)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // unsigned int warpID = thID >> 5;
        unsigned int ID = 0;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int IID = __ldg(&neworder_query->OL_I_ID[NEWORDER_CNT * ID + NO]);
            unsigned int SWID = __ldg(&neworder_query->OL_SUPPLY_W_ID[NEWORDER_CNT * ID + NO]);
            unsigned int Loc = SWID * 100000 + IID;
            unsigned int TID = NO;
            stock_access->S_ID[NEWORDER_CNT * (20 + ID) + NO] = Loc;
            stock_access->TID[NO] = TID;
            atomicAdd(&stock_auxiliary->HIST[Loc], 1);
        }
    }
    __device__ void get_orderline_access(unsigned int NO,
                                         NEWORDER_QUERY *neworder_query,
                                         ORDERLINE_ACCESS *orderline_access,
                                         ORDERLINE_AUXILIARY *orderline_auxiliary)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // unsigned int warpID = thID >> 5;
        unsigned int ID = 0;
        unsigned int O_OL_CNT = __ldg(&neworder_query->O_OL_CNT[NO]);
        if (ID < O_OL_CNT)
        {
            unsigned int WID = __ldg(&neworder_query->W_ID[NO]);
            unsigned int OOLID = __ldg(&neworder_query->O_OL_ID[NO]);
            unsigned int Loc = WID * 450000 + OOLID + ID;
            unsigned int TID = NO;
            orderline_access->OL_ID[NEWORDER_CNT * (35 + ID) + NO] = Loc;
            orderline_access->TID[NO] = TID;
            atomicAdd(&orderline_auxiliary->HIST[Loc], 1);
        }
    }
    __device__ void get_Auxiliary_Structure(unsigned int NO,
                                            NEWORDER_QUERY *neworder_query,
                                            NEWORDERQUERY_ACCESS *neworderquery_access,
                                            NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary)

    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (thID / NEWORDER_CNT == 0)
        {
            get_warehouse_access(NO, neworder_query, &neworderquery_access->warehouse_access, &neworderquery_auxiliary->warehouse_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 1)
        {
            get_district_access(NO, neworder_query, &neworderquery_access->district_access, &neworderquery_auxiliary->distrct_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 2)
        {
            get_customer_access(NO, neworder_query, &neworderquery_access->customer_access, &neworderquery_auxiliary->customer_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 3)
        {
            get_neworder_access(NO, neworder_query, &neworderquery_access->neworder_access, &neworderquery_auxiliary->neworder_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 4)
        {
            get_order_access(NO, neworder_query, &neworderquery_access->order_access, &neworderquery_auxiliary->order_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 5)
        {
            get_item_access(NO, neworder_query, &neworderquery_access->item_access, &neworderquery_auxiliary->item_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 6)
        {
            get_stock_access(NO, neworder_query, &neworderquery_access->stock_access, &neworderquery_auxiliary->stock_auxiliary);
        }
        else if (thID / NEWORDER_CNT == 7)
        {
            get_orderline_access(NO, neworder_query, &neworderquery_access->orderline_access, &neworderquery_auxiliary->orderline_auxiliary);
        }
    }
    __device__ void calculate_prefix_offset(NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary)
    {
        unsigned int blockID = blockIdx.x;
        if (blockID == 0)
        {
            prefix_sum_in_block(neworderquery_auxiliary->warehouse_auxiliary.HIST, WAREHOUSE_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->warehouse_auxiliary.HIST, WAREHOUSE_SIZE);
        }
        else if (blockID == 1)
        {
            prefix_sum_in_block(neworderquery_auxiliary->distrct_auxiliary.HIST, DISTRICT_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->distrct_auxiliary.HIST, DISTRICT_SIZE);
        }
        else if (blockID == 2)
        {
            prefix_sum_in_block(neworderquery_auxiliary->customer_auxiliary.HIST, CUSTOMER_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->customer_auxiliary.HIST, CUSTOMER_SIZE);
        }
        else if (blockID == 3)
        {
            prefix_sum_in_block(neworderquery_auxiliary->neworder_auxiliary.HIST, NEWORDER_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->neworder_auxiliary.HIST, NEWORDER_SIZE);
        }
        else if (blockID == 4)
        {
            prefix_sum_in_block(neworderquery_auxiliary->order_auxiliary.HIST, ORDER_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->order_auxiliary.HIST, ORDER_SIZE);
        }
        else if (blockID == 5)
        {
            prefix_sum_in_block(neworderquery_auxiliary->item_auxiliary.HIST, ITEM_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->item_auxiliary.HIST, ITEM_SIZE);
        }
        else if (blockID == 6)
        {
            prefix_sum_in_block(neworderquery_auxiliary->stock_auxiliary.HIST, STOCK_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->stock_auxiliary.HIST, STOCK_SIZE);
        }
        else if (blockID == 7)
        {
            prefix_sum_in_block(neworderquery_auxiliary->orderline_auxiliary.HIST, ORDERLINE_SIZE);
            __syncthreads();
            prefix_sum_between_block(neworderquery_auxiliary->orderline_auxiliary.HIST, ORDERLINE_SIZE);
        }
    }
    __device__ void execute_warehouse(unsigned int NO,
                                      NEWORDER_SET *neworder_set,
                                      NEWORDER_QUERY *neworder_query,
                                      WAREHOUSE_ACCESS *warehouse_access,
                                      WAREHOUSE_AUXILIARY *warehouse_auxiliary,
                                      SNAPSHOT *snapshot)
    {
        // unsigned int offset = warehouse_auxiliary->OFFSET[NO];
        unsigned int hist = warehouse_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < WAREHOUSE_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = warehouse_auxiliary->LOCK[NO];
                __READ(lock, 0, 1, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, neworder_set->set_local_set, snapshot->warehouse_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_district(unsigned int NO,
                                     NEWORDER_SET *neworder_set,
                                     NEWORDER_QUERY *neworder_query,
                                     DISTRICT_ACCESS *district_access,
                                     DISTRICT_AUXILIARY *district_auxiliary,
                                     SNAPSHOT *snapshot)
    {
        // unsigned int offset = district_auxiliary->OFFSET[NO];
        unsigned int hist = district_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < DISTRICT_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = district_auxiliary->LOCK[NO];
                __READ(lock, 1, 2, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, neworder_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 1, 4, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, neworder_set->set_local_set, snapshot->district_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_customer(unsigned int NO,
                                     NEWORDER_SET *neworder_set,
                                     NEWORDER_QUERY *neworder_query,
                                     CUSTOMER_ACCESS *customer_access,
                                     CUSTOMER_AUXILIARY *customer_auxiliary,
                                     SNAPSHOT *snapshot)
    {
        // unsigned int offset = customer_auxiliary->OFFSET[NO];
        unsigned int hist = customer_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < CUSTOMER_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = customer_auxiliary->LOCK[NO];
                __READ(lock, 2, 4, NO, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 2, 10, NO, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 2, 13, NO, NEWORDER_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, neworder_set->set_local_set, snapshot->customer_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_neworder(unsigned int NO,
                                     NEWORDER_SET *neworder_set,
                                     NEWORDER_QUERY *neworder_query,
                                     NEWORDER_ACCESS *neworder_access,
                                     NEWORDER_AUXILIARY *neworder_auxiliary,
                                     SNAPSHOT *snapshot)
    {
        // unsigned int offset = neworder_auxiliary->OFFSET[NO];
        unsigned int hist = neworder_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < NEWORDER_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = neworder_auxiliary->LOCK[NO];
                __WRITE(lock, 3, 0, NO, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 3, 1, NO, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 3, 2, NO, NEWORDER_CNT, NEWORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->neworder_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_order(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  ORDER_ACCESS *order_access,
                                  ORDER_AUXILIARY *order_auxiliary,
                                  SNAPSHOT *snapshot)
    {
        // unsigned int offset = order_auxiliary->OFFSET[NO];
        unsigned int hist = order_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < ORDER_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = order_auxiliary->LOCK[NO];
                __WRITE(lock, 4, 0, NO, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
                __WRITE(lock, 4, 1, NO, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
                __WRITE(lock, 4, 2, NO, NEWORDER_CNT, ORDER_SIZE, ORDER_COLUMN, 0, neworder_set->set_local_set, snapshot->order_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_item(unsigned int NO,
                                 NEWORDER_SET *neworder_set,
                                 NEWORDER_QUERY *neworder_query,
                                 ITEM_ACCESS *item_access,
                                 ITEM_AUXILIARY *item_auxiliary,
                                 SNAPSHOT *snapshot)
    {
        // unsigned int offset = item_auxiliary->OFFSET[NO];
        unsigned int hist = item_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < ITEM_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = item_auxiliary->LOCK[NO];
                __READ(lock, 5, 2, NO, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
                __READ(lock, 5, 3, NO, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
                __READ(lock, 5, 4, NO, NEWORDER_CNT, ITEM_SIZE, ITEM_COLUMN, 0, neworder_set->set_local_set, snapshot->item_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_stock(unsigned int NO,
                                  NEWORDER_SET *neworder_set,
                                  NEWORDER_QUERY *neworder_query,
                                  STOCK_ACCESS *stock_access,
                                  STOCK_AUXILIARY *stock_auxiliary,
                                  SNAPSHOT *snapshot)
    {
        // unsigned int offset = stock_auxiliary->OFFSET[NO];
        unsigned int hist = stock_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < STOCK_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = stock_auxiliary->LOCK[NO];
                __WRITE(lock, 20, 2, NO, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
                __WRITE(lock, 20, 3, NO, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
                __WRITE(lock, 20, 4, NO, NEWORDER_CNT, STOCK_SIZE, STOCK_COLUMN, 0, neworder_set->set_local_set, snapshot->stock_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_orderline(unsigned int NO,
                                      NEWORDER_SET *neworder_set,
                                      NEWORDER_QUERY *neworder_query,
                                      ORDERLINE_ACCESS *orderline_access,
                                      ORDERLINE_AUXILIARY *orderline_auxiliary,
                                      SNAPSHOT *snapshot)
    {
        // unsigned int offset = orderline_auxiliary->OFFSET[NO];
        unsigned int hist = orderline_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < ORDERLINE_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = orderline_auxiliary->LOCK[NO];
                __WRITE(lock, 35, 1, NO, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
                __WRITE(lock, 35, 2, NO, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
                __WRITE(lock, 35, 3, NO, NEWORDER_CNT, ORDERLINE_SIZE, ORDERLINE_COLUMN, 0, neworder_set->set_local_set, snapshot->orderline_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute(unsigned int NO,
                            NEWORDER_SET *neworder_set,
                            NEWORDER_QUERY *neworder_query,
                            NEWORDERQUERY_ACCESS *neworderquery_access,
                            NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary,
                            SNAPSHOT *snapshot)
    {
        if (NO < WAREHOUSE_SIZE)
        {
            execute_warehouse(NO, neworder_set, neworder_query, &neworderquery_access->warehouse_access, &neworderquery_auxiliary->warehouse_auxiliary, snapshot);
        }
        if (NO < DISTRICT_SIZE)
        {
            execute_district(NO, neworder_set, neworder_query, &neworderquery_access->district_access, &neworderquery_auxiliary->distrct_auxiliary, snapshot);
        }
        if (NO < CUSTOMER_SIZE)
        {
            execute_customer(NO, neworder_set, neworder_query, &neworderquery_access->customer_access, &neworderquery_auxiliary->customer_auxiliary, snapshot);
        }
        if (NO < NEWORDER_SIZE)
        {
            execute_neworder(NO, neworder_set, neworder_query, &neworderquery_access->neworder_access, &neworderquery_auxiliary->neworder_auxiliary, snapshot);
        }
        if (NO < ORDER_SIZE)
        {
            execute_order(NO, neworder_set, neworder_query, &neworderquery_access->order_access, &neworderquery_auxiliary->order_auxiliary, snapshot);
        }
        if (NO < ITEM_SIZE)
        {
            execute_item(NO, neworder_set, neworder_query, &neworderquery_access->item_access, &neworderquery_auxiliary->item_auxiliary, snapshot);
        }
        if (NO < STOCK_SIZE)
        {
            execute_stock(NO, neworder_set, neworder_query, &neworderquery_access->stock_access, &neworderquery_auxiliary->stock_auxiliary, snapshot);
        }
        if (NO < ORDERLINE_SIZE)
        {
            execute_orderline(NO, neworder_set, neworder_query, &neworderquery_access->orderline_access, &neworderquery_auxiliary->orderline_auxiliary, snapshot);
        }
    }
}

namespace GACCO_PAYMENT_NAMESPACE
{
    __device__ void get_warehouse_access(unsigned int NO,
                                         PAYMENT_QUERY *payment_query,
                                         WAREHOUSE_ACCESS *warehouse_access,
                                         WAREHOUSE_AUXILIARY *warehouse_auxiliary)
    {
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int Loc = WID;
        unsigned int TID = NO;
        warehouse_access->W_ID[NO] = Loc;
        warehouse_access->TID[NO] = TID;
        atomicAdd(&warehouse_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_district_access(unsigned int NO,
                                        PAYMENT_QUERY *payment_query,
                                        DISTRICT_ACCESS *district_access,
                                        DISTRICT_AUXILIARY *district_auxiliary)
    {
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int Loc = WID * 10 + DID;
        unsigned int TID = NO;
        district_access->D_ID[NO] = Loc;
        district_access->TID[NO] = TID;
        atomicAdd(&district_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_customer_access(unsigned int NO,
                                        PAYMENT_QUERY *payment_query,
                                        CUSTOMER_ACCESS *customer_access,
                                        CUSTOMER_AUXILIARY *customer_auxiliary)
    {
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int DID = __ldg(&payment_query->D_ID[NO]);
        unsigned int CID = __ldg(&payment_query->C_ID[NO]);
        unsigned int Loc = (WID * 10 + DID) * 3000 + CID;
        unsigned int TID = NO;
        customer_access->C_ID[NO] = Loc;
        customer_access->TID[NO] = TID;
        atomicAdd(&customer_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_history_access(unsigned int NO,
                                       PAYMENT_QUERY *payment_query,
                                       HISTORY_ACCESS *history_access,
                                       HISTORY_AUXILIARY *history_auxiliary)
    {
        unsigned int WID = __ldg(&payment_query->W_ID[NO]);
        unsigned int HID = __ldg(&payment_query->H_ID[NO]);
        unsigned int Loc = WID * 30000 + HID;
        unsigned int TID = NO;
        history_access->H_ID[NO] = Loc;
        history_access->TID[NO] = TID;
        atomicAdd(&history_auxiliary->HIST[Loc], 1);
    }
    __device__ void get_Auxiliary_Structure(unsigned int NO,
                                            PAYMENT_QUERY *payment_query,
                                            PAYMENTQUERY_ACCESS *paymentquery_access,
                                            PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (thID / PAYMENT_CNT == 0)
        {
            get_warehouse_access(NO, payment_query, &paymentquery_access->warehouse_access, &paymentquery_auxiliary->warehouse_auxiliary);
        }
        else if (thID / PAYMENT_CNT == 1)
        {
            get_district_access(NO, payment_query, &paymentquery_access->district_access, &paymentquery_auxiliary->distrct_auxiliary);
        }
        else if (thID / PAYMENT_CNT == 2)
        {
            get_customer_access(NO, payment_query, &paymentquery_access->customer_access, &paymentquery_auxiliary->customer_auxiliary);
        }
        else if (thID / PAYMENT_CNT == 3)
        {
            get_history_access(NO, payment_query, &paymentquery_access->history_access, &paymentquery_auxiliary->history_auxiliary);
        }
    }
    __device__ void calculate_prefix_offset(PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary)
    {
        unsigned int blockID = blockIdx.x;
        if (blockID == 0)
        {
            prefix_sum_in_block(paymentquery_auxiliary->warehouse_auxiliary.HIST, WAREHOUSE_SIZE);
            __syncthreads();
            prefix_sum_between_block(paymentquery_auxiliary->warehouse_auxiliary.HIST, WAREHOUSE_SIZE);
        }
        else if (blockID == 1)
        {
            prefix_sum_in_block(paymentquery_auxiliary->distrct_auxiliary.HIST, DISTRICT_SIZE);
            __syncthreads();
            prefix_sum_between_block(paymentquery_auxiliary->distrct_auxiliary.HIST, DISTRICT_SIZE);
        }
        else if (blockID == 2)
        {
            prefix_sum_in_block(paymentquery_auxiliary->customer_auxiliary.HIST, CUSTOMER_SIZE);
            __syncthreads();
            prefix_sum_between_block(paymentquery_auxiliary->customer_auxiliary.HIST, CUSTOMER_SIZE);
        }
        else if (blockID == 3)
        {
            prefix_sum_in_block(paymentquery_auxiliary->history_auxiliary.HIST, HISTORY_SIZE);
            __syncthreads();
            prefix_sum_between_block(paymentquery_auxiliary->history_auxiliary.HIST, HISTORY_SIZE);
        }
    }
    __device__ void execute_warehouse(unsigned int NO,
                                      PAYMENT_SET *payment_set,
                                      PAYMENT_QUERY *payment_query,
                                      WAREHOUSE_ACCESS *warehouse_access,
                                      WAREHOUSE_AUXILIARY *warehouse_auxiliary,
                                      SNAPSHOT *snapshot)
    {
        // unsigned int offset = warehouse_auxiliary->OFFSET[NO];
        unsigned int hist = warehouse_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < WAREHOUSE_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = warehouse_auxiliary->LOCK[NO];
                __READ(lock, 0, 3, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __READ(lock, 0, 4, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __READ(lock, 0, 5, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __READ(lock, 0, 6, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __READ(lock, 0, 7, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __READ(lock, 0, 8, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                __WRITE(lock, 1, 2, NO, NEWORDER_CNT, WAREHOUSE_SIZE, WAREHOUSE_COLUMN, 0, payment_set->set_local_set, snapshot->warehouse_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_district(unsigned int NO,
                                     PAYMENT_SET *payment_set,
                                     PAYMENT_QUERY *payment_query,
                                     DISTRICT_ACCESS *district_access,
                                     DISTRICT_AUXILIARY *district_auxiliary,
                                     SNAPSHOT *snapshot)
    {
        // unsigned int offset = district_auxiliary->OFFSET[NO];
        unsigned int hist = district_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < DISTRICT_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = district_auxiliary->LOCK[NO];
                __READ(lock, 2, 5, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 2, 6, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 2, 7, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 2, 8, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 2, 9, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __READ(lock, 2, 10, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                __WRITE(lock, 3, 3, NO, NEWORDER_CNT, DISTRICT_SIZE, DISTRICT_COLUMN, 0, payment_set->set_local_set, snapshot->district_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_customer(unsigned int NO,
                                     PAYMENT_SET *payment_set,
                                     PAYMENT_QUERY *payment_query,
                                     CUSTOMER_ACCESS *customer_access,
                                     CUSTOMER_AUXILIARY *customer_auxiliary,
                                     SNAPSHOT *snapshot)
    {
        // unsigned int offset = customer_auxiliary->OFFSET[NO];
        unsigned int hist = customer_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < CUSTOMER_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = customer_auxiliary->LOCK[NO];
                __READ(lock, 4, 3, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 4, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 11, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 12, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 13, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 14, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 15, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 16, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 17, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 18, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 19, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __READ(lock, 4, 20, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __WRITE(lock, 5, 5, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                __WRITE(lock, 5, 9, NO, PAYMENT_CNT, CUSTOMER_SIZE, CUSTOMER_COLUMN, 0, payment_set->set_local_set, snapshot->customer_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute_history(unsigned int NO,
                                    PAYMENT_SET *payment_set,
                                    PAYMENT_QUERY *payment_query,
                                    HISTORY_ACCESS *history_access,
                                    HISTORY_AUXILIARY *history_auxiliary,
                                    SNAPSHOT *snapshot)
    {
        // unsigned int offset = history_auxiliary->OFFSET[NO];
        unsigned int hist = history_auxiliary->HIST[NO];
        unsigned int count = 0;
        if (NO < CUSTOMER_SIZE)
        {
            while (count != hist)
            {
                unsigned int lock = history_auxiliary->LOCK[NO];
                __WRITE(lock, 6, 0, NO, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 6, 1, NO, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 6, 2, NO, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 6, 3, NO, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->neworder_snapshot);
                __WRITE(lock, 6, 4, NO, PAYMENT_CNT, HISTORY_SIZE, HISTORY_COLUMN, 0, payment_set->set_local_set, snapshot->neworder_snapshot);
                count += 1;
            }
        }
    }
    __device__ void execute(unsigned int NO,
                            PAYMENT_SET *payment_set,
                            PAYMENT_QUERY *payment_query,
                            PAYMENTQUERY_ACCESS *paymentquery_access,
                            PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary,
                            SNAPSHOT *snapshot)
    {
        if (NO < WAREHOUSE_SIZE)
        {
            execute_warehouse(NO, payment_set, payment_query, &paymentquery_access->warehouse_access, &paymentquery_auxiliary->warehouse_auxiliary, snapshot);
        }
        if (NO < DISTRICT_SIZE)
        {
            execute_district(NO, payment_set, payment_query, &paymentquery_access->district_access, &paymentquery_auxiliary->distrct_auxiliary, snapshot);
        }
        if (NO < CUSTOMER_SIZE)
        {
            execute_customer(NO, payment_set, payment_query, &paymentquery_access->customer_access, &paymentquery_auxiliary->customer_auxiliary, snapshot);
        }
        if (NO < HISTORY_SIZE)
        {
            execute_history(NO, payment_set, payment_query, &paymentquery_access->history_access, &paymentquery_auxiliary->history_auxiliary, snapshot);
        }
    }
}
