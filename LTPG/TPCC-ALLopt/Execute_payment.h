#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include "Query.h"
#include "Database.h"
#include "Genericfunction.h"

namespace PAYMENT_NAMESPACE
{

    __device__ void get_c_id(unsigned int NO,
                             PAYMENT_SET *payment_set,
                             PAYMENT_QUERY *payment_query,
                             LOG *log,
                             SNAPSHOT *snapshot,
                             INDEX *index)
    { // get C_ID from C_NAME_INDEX
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

    /// regist the Read operation on the data item with KEY (W_ID) to table LOG_WAREHOUSE_R
    /// select warehouse
    ///
    /// @param NO                the NO instance of this transaction
    /// @param payment_set      local read/write set
    /// @param payment_query    query info
    /// @param log
    /// @param snapshot
    /// @param index
    ///
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

    /// read the W_NAME (column 3), W_STREET_1 (column 4), W_STREET_2 (column 5) with KEY (W_ID) from warehouse_snapshot to local set
    /// select warehouse
    ///
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

    /// read the W_CITY (column 6), W_STATE (column 7), W_ZIP (column 8) with KEY (W_ID) from warehouse_snapshot to local set
    /// select warehouse
    ///
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

    /// regist the update operation on the data item to table LOG_warehouse_R
    /// update warehouse
    /// FOR REDUCE_UPGRADE: store the H_AMOUNT in log->W_YTD
    ///
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

    /// regist the update operation on the data item to table LOG_district_R
    /// update district
    /// FOR REDUCE_UPGRADE: store the H_AMOUNT in log->D_YTD
    ///
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

    ///
    /// Assign execute tasks to warp
    /// Each operation step occupy 1 warp
    ///
    __device__ void execute(unsigned int NO,
                            PAYMENT_SET *payment_set,
                            PAYMENT_QUERY *payment_query,
                            LOG *log,
                            SNAPSHOT *snapshot,
                            INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        wID = wID % EXECUTE_WARP;
        if (NO < PAYMENT_CNT)
        {
            if (wID == 130)
            {
                op_0_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 131)
            {
                op_0_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 132)
            {
                op_0_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 133)
            {
                op_1_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 134)
            {
                op_2_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 135)
            {
                op_2_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 136)
            {
                op_2_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 137)
            {
                op_3_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 138)
            {
                op_4_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 139)
            {
                op_4_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 140)
            {
                op_4_2(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 141)
            {
                op_4_3(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 142)
            {
                op_4_4(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 143)
            {
                op_5_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 144)
            {
                op_5_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 145)
            {
                op_6_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 146)
            {
                op_6_1(NO, payment_set, payment_query, log, snapshot, index);
            }
        }
    }

    // check conflict in
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

    ///
    /// Assign check tasks to warp
    /// read operation need to check raw conflict , write operation need to check waw and war conflict.
    /// Each operation step occupy 1 warp
    ///
    __device__ void check(unsigned int NO,
                          PAYMENT_SET *payment_set,
                          PAYMENT_QUERY *payment_query,
                          LOG *log,
                          SNAPSHOT *snapshot,
                          INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        wID = wID % CHECK_WARP;
        if (NO < PAYMENT_CNT)
        {
            if (wID == 82)
            {
                check_0_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 83)
            {
                check_1_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 84)
            {
                check_1_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 85)
            {
                check_2_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 86)
            {
                check_3_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 87)
            {
                check_3_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 88)
            {
                check_4_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 89)
            {
                check_5_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 90)
            {
                check_5_1(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 91)
            {
                check_6_0(NO, payment_set, payment_query, log, snapshot, index);
            }
            else if (wID == 92)
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

    /// W_YTD value and Loc is stored in Log
    ///
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

    ///
    /// Assign write_back tasks to warp
    /// write operations can be written back (commit) only if waw conflicts do not exist and raw and war conflicts do not exist at the same time.
    /// op_5 , op_6_x  :  Each operation step occupy 1 warp
    ///
    __device__ void write_back(unsigned int NO,
                               PAYMENT_SET *payment_set,
                               PAYMENT_QUERY *payment_query,
                               LOG *log,
                               SNAPSHOT *snapshot,
                               INDEX *index)
    {
        unsigned int thID = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int wID = thID >> 5;
        wID = wID % WRITEBACK_WARP;
        if (NO < PAYMENT_CNT)
        {
            if (wID == 63)
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
                    if (wID == 64)
                    {
                        writeback_6_0(NO, payment_set, payment_query, log, snapshot, index);
                    }
                    else if (wID == 65)
                    {
                        writeback_6_1(NO, payment_set, payment_query, log, snapshot, index);
                    }
                    // else if (wID == 66)
                    // {
                    //     // writeback_1_0(NO, payment_set, payment_query, log, snapshot, index);
                    // }
                    // else if (wID == 67)
                    // {
                    //     // writeback_3_0(NO, payment_set, payment_query, log, snapshot, index);
                    // }
                    else if (wID == 66)
                    {
                        writeback_5_0(NO, payment_set, payment_query, log, snapshot, index);
                    }
                }
            }
        }
    }

    /// only called once by KERNEL_WRITEBACK
    /// reduce_update optimization for updating warehouse table and district table in payment transaction
    /// no need to use warp partitioning
    ///
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
