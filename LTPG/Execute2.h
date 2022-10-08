#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <thrust/extrema.h>
#include "Query.h"
#include "Database.h"

using namespace cooperative_groups;

// change hash bucket size, const cooperative groups

struct SET
{
    // int32_t WID;
    int32_t Loc;
    GLOBAL local_set;
};

struct FLAG
{
    int R_CNT = 0;
    int W_CNT = 0;
    // int R_MIN = 1 << 31 - 1;
    // int W_MIN = 1 << 31 - 1;
    int lock_R[NORMAL_HASH_SIZE]; // 1 unlock, 0 lock
    int TID_LIST_R[NORMAL_HASH_SIZE];
    int lock_W[NORMAL_HASH_SIZE];
    int TID_LIST_W[NORMAL_HASH_SIZE];
};

struct FLAGCOMP
{
    int R_CNT = 0;
    int W_CNT = 0;
    // int R_MIN = 1 << 31 - 1;
    // int W_MIN = 1 << 31 - 1;
    int lock_R[COMPETITION_HASH_SIZE]; // 1 unlock, 0 lock
    int TID_LIST_R[COMPETITION_HASH_SIZE];
    int lock_W[COMPETITION_HASH_SIZE];
    int TID_LIST_W[COMPETITION_HASH_SIZE];
};
struct Set_n
{
    int64_t TID;
    NewOrderQuery query;
    SET set[50];
    int war = 0;
    int raw = 0;
    int waw = 0;
};
struct Set_p
{
    int64_t TID;
    Payment query;
    SET set[7];
    int war = 0;
    int raw = 0;
    int waw = 0;
};

__device__ int hash(int loc, int htblsize, FLAGCOMP *flagcomp)
{
    int htblloc = loc % htblsize;
    // if (loc > WAREHOUSE_SIZE + DISTRICT_SIZE)
    // {
    //     bool block = true;
    //     while (block)
    //     {
    //         if (atomicCAS(&flagcomp[htblloc], (1 << 31 - 1), 0) == 1)
    //         {
    //             int tmp = flagcomp[htblloc];
    //             if (tmp == 1 << 31 - 1 || tmp == loc)
    //             {
    //                 atomicExch(&flagcomp[htblloc].loc, loc);
    //                 atomicExch(&flagcomp[htblloc].lock_loc, 1);
    //                 block = false;
    //             }
    //             else
    //             {
    //                 htblloc += 1;
    //                 if (htblloc == htblsize)
    //                 {
    //                     htblloc = WAREHOUSE_SIZE + DISTRICT_SIZE;
    //                 }
    //             }
    //         }
    //     }
    // }
    // else
    // {
    //     htblloc = loc;
    // }
    htblloc = loc;
    return htblloc;
}

__device__ int findhash(int loc, int htblsize, FLAGCOMP *flagcomp)
{
    int htblloc = loc % htblsize;
    // if (loc > WAREHOUSE_SIZE + DISTRICT_SIZE)
    // {
    //     bool block = true;
    //     while (block)
    //     {
    //         int tmp = flagcomp[htblloc].loc;
    //         if (tmp == loc)
    //         {
    //             block = false;
    //         }
    //         else
    //         {
    //             htblloc += 1;
    //             if (htblloc == htblsize)
    //             {
    //                 htblloc = WAREHOUSE_SIZE + DISTRICT_SIZE;
    //             }
    //         }
    //     }
    // }
    // else
    // {
    //     htblloc = loc;
    // }
    htblloc = loc;
    return htblloc;
}

namespace NEWORDER_QUERY
{
    int n_op_size = 8;

    __device__ void op_n_0(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from warehouse
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_n_d[0] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = neworder_d[NO].TID;
        int WID = neworder_d[NO].query.W_ID;
        // int O_OL_CNT = neworder_d[NO].query.O_OL_CNT;
        // neworder_d[NO].set[0].WID = WID;
        int Loc = WID;
        neworder_d[NO].set[0].Loc = Loc;
        // printf("neworder_d[NO].set[0].Loc = %d\n", neworder_d[NO].set[0].Loc);
        GLOBAL temp = current_2D_d[Loc];
        neworder_d[NO].set[0].local_set.data[1] = temp.data[1];
        // printf("neworder_d[NO].set[0].local_set.data[1] = %d\n", neworder_d[NO].set[0].local_set.data[1]);

        // neworder_d[NO].update_cnt = 1 + 2 * O_OL_CNT;
        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].R_CNT, 1);
        atomicAdd(&flagcomp[htblloc].R_CNT, 1);
        bool blocked = true;
        // printf("start %d:0\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:0\n", NO);
                atomicMin(&flagcomp[htblloc].TID_LIST_R[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:0\n", NO);
            }
        }
        // printf("end %d:0\n", NO);
    }

    __device__ void op_n_1(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from district
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_n_d[1] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = neworder_d[NO].TID;
        int WID = neworder_d[NO].query.W_ID;
        int DID = neworder_d[NO].query.D_ID;
        // neworder_d[NO].set[1].WID = WID;
        int Loc = WAREHOUSE_SIZE + WID * 10 + DID;
        neworder_d[NO].set[1].Loc = Loc;
        // printf("neworder_d[%d].set[1].Loc = %d\n", NO, neworder_d[NO].set[1].Loc);
        GLOBAL temp = current_2D_d[Loc];

        neworder_d[NO].set[1].local_set.data[2] = temp.data[2];
        neworder_d[NO].set[1].local_set.data[4] = temp.data[4];
        // printf("neworder_d[NO].set[1].local_set.data[2] = %d\n", neworder_d[NO].set[1].local_set.data[2]);
        // printf("neworder_d[NO].set[1].local_set.data[4] = %d\n", neworder_d[NO].set[1].local_set.data[4]);
        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].R_CNT, 1);
        atomicAdd(&flagcomp[htblloc].R_CNT, 1);

        bool blocked = true;
        // printf("start %d:1\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:1\n", NO);
                atomicMin(&flagcomp[htblloc].TID_LIST_R[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:1\n", NO);
            }
        }
        // printf("end %d:1\n", NO);
    }

    __device__ void op_n_2(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from customer
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_n_d[2] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = neworder_d[NO].TID;
        int WID = neworder_d[NO].query.W_ID;
        int DID = neworder_d[NO].query.D_ID;
        int CID = neworder_d[NO].query.C_ID;
        // neworder_d[NO].set[2].WID = WID;

        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + WID * 30000 + 3000 * DID + CID;
        neworder_d[NO].set[2].Loc = Loc;
        // printf("neworder_d[NO].set[2].Loc = %d\n", neworder_d[NO].set[2].Loc);
        GLOBAL temp = current_2D_d[Loc];

        neworder_d[NO].set[2].local_set.data[4] = temp.data[4];
        neworder_d[NO].set[2].local_set.data[10] = temp.data[10];
        neworder_d[NO].set[2].local_set.data[13] = temp.data[13];
        // printf("neworder_d[NO].set[2].local_set.data[4] = %d\n", neworder_d[NO].set[2].local_set.data[4]);
        // printf("neworder_d[NO].set[2].local_set.data[10] = %d\n", neworder_d[NO].set[2].local_set.data[10]);
        // printf("neworder_d[NO].set[2].local_set.data[13] = %d\n", neworder_d[NO].set[2].local_set.data[13]);
        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].R_CNT, 1);
        bool blocked = true;
        // printf("start %d:2\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_R[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:2\n", NO);
                atomicMin(&flag[htblloc].TID_LIST_R[TID % NORMAL_HASH_SIZE], TID);
                atomicExch(&flag[htblloc].lock_R[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:2\n", NO);
            }
        }
        // printf("end %d:2\n", NO);
    }

    __device__ void op_n_3(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // insert neworder
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_n_d[3] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = neworder_d[NO].TID;
        int NOID = neworder_d[NO].query.N_O_ID;
        int WID = neworder_d[NO].query.W_ID;
        int DID = neworder_d[NO].query.D_ID;
        // neworder_d[NO].set[3].WID = WID;
        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + WID * 30000 + NOID; // 60011 + NOID;
        neworder_d[NO].set[3].Loc = Loc;
        // printf("neworder_d[NO].set[3].Loc = %d\n", neworder_d[NO].set[3].Loc);
        // printf("%d : %d\n", NO, NOID);

        neworder_d[NO].set[3].local_set.data[0] = NOID;
        neworder_d[NO].set[3].local_set.data[1] = DID;
        neworder_d[NO].set[3].local_set.data[2] = WID;
        // printf("neworder_d[NO].set[3].local_set.data[0] = %d\n", neworder_d[NO].set[3].local_set.data[0]);
        // printf("neworder_d[NO].set[3].local_set.data[1] = %d\n", neworder_d[NO].set[3].local_set.data[1]);
        // printf("neworder_d[NO].set[3].local_set.data[2] = %d\n", neworder_d[NO].set[3].local_set.data[2]);
        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);
        bool blocked = true;
        // printf("start %d:3\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:3\n", NO);
                atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:3\n", NO);
            }
        }
        // printf("end %d:3\n", NO);
    }

    __device__ void op_n_4(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // insert order
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_n_d[4] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = neworder_d[NO].TID;
        int OID = neworder_d[NO].query.O_ID;
        int DID = neworder_d[NO].query.D_ID;
        int WID = neworder_d[NO].query.W_ID;
        // neworder_d[NO].set[4].WID = WID;

        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + WID * 30000 + OID;
        neworder_d[NO].set[4].Loc = Loc;

        // printf("neworder_d[NO].set[4].Loc = %d\n", neworder_d[NO].set[4].Loc);

        neworder_d[NO].set[4].local_set.data[0] = OID;
        neworder_d[NO].set[4].local_set.data[1] = DID;
        neworder_d[NO].set[4].local_set.data[2] = WID;
        // printf("neworder_d[NO].set[4].local_set.data[0] = %d\n", neworder_d[NO].set[4].local_set.data[0]);
        // printf("neworder_d[NO].set[4].local_set.data[1] = %d\n", neworder_d[NO].set[4].local_set.data[1]);
        // printf("neworder_d[NO].set[4].local_set.data[2] = %d\n", neworder_d[NO].set[4].local_set.data[2]);
        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);
        bool blocked = true;
        // printf("start %d:4\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:4\n", NO);
                atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:4\n", NO);
            }
        }
        // printf("end %d:4\n", NO);
    }

    __device__ void op_n_5(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from item loop
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        int O_OL_CNT = neworder_d[NO].query.O_OL_CNT;
        // if (OLID == 0)
        // {
        //     printf("%d O_OL_CNT : %d\n", NO, O_OL_CNT);
        // }
        if (OLID < O_OL_CNT)
        {
            // printf("op_n_d[%d] wID : %d\t thID : %d\t\n", 5 + OLID, wID, thID);

            int TID = neworder_d[NO].TID;
            int IID = neworder_d[NO].query.INFO[OLID].OL_I_ID;
            // neworder_d[NO].set[5 + OLID].WID = WAREHOUSE_SIZE;
            int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + IID;
            neworder_d[NO].set[5 + OLID].Loc = Loc;
            // printf("neworder_d[NO].set[5 + %d].Loc = %d\n", OLID, neworder_d[NO].set[5 + OLID].Loc);
            GLOBAL temp = current_2D_d[Loc];

            neworder_d[NO].set[5 + OLID].local_set.data[2] = temp.data[2];
            neworder_d[NO].set[5 + OLID].local_set.data[3] = temp.data[3];
            neworder_d[NO].set[5 + OLID].local_set.data[4] = temp.data[4];
            // printf("neworder_d[NO].set[5 + OLID].local_set.data[2] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[2]);
            // printf("neworder_d[NO].set[5 + OLID].local_set.data[3] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[3]);
            // printf("neworder_d[NO].set[5 + OLID].local_set.data[4] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[4]);
            int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

            atomicAdd(&flag[htblloc].R_CNT, 1);
            bool blocked = true;
            // printf("start %d:%d\n", 6, OLID);
            while (blocked)
            {
                if (atomicCAS(&flag[htblloc].lock_R[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
                {
                    // printf("lock %d:%d\n", 6, OLID);
                    atomicMin(&flag[htblloc].TID_LIST_R[TID % NORMAL_HASH_SIZE], TID);
                    atomicExch(&flag[htblloc].lock_R[TID % NORMAL_HASH_SIZE], 1);
                    blocked = false;
                    // printf("unlock %d:%d\n", 6, OLID);
                }
            }
            // printf("end %d:%d\n", 6, OLID);
        }
    }

    __device__ void op_n_6(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // update stock loop
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        int O_OL_CNT = neworder_d[NO].query.O_OL_CNT;
        if (OLID < O_OL_CNT)
        {
            // printf("op_n_d[%d] wID : %d\t thID : %d\t\n", 20 + OLID, wID, thID);

            int TID = neworder_d[NO].TID;
            int SWID = neworder_d[NO].query.INFO[OLID].OL_SUPPLY_W_ID;
            int SID = neworder_d[NO].query.INFO[OLID].OL_I_ID;
            int quantity = neworder_d[NO].query.INFO[OLID].OL_QUANTITY;
            int price = current_2D_d[TABLE_SIZE_1D - ITEM_SIZE + SID].data[2];
            // neworder_d[NO].set[20 + OLID].WID = SWID;
            int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + (SWID * 100000) + SID;
            neworder_d[NO].set[20 + OLID].Loc = Loc;
            // printf("neworder_d[%d].set[20 + %d].Loc = %d\n", NO, OLID, SID);
            GLOBAL temp = current_2D_d[Loc];

            neworder_d[NO].set[20 + OLID].local_set.data[2] = temp.data[2] - quantity;
            neworder_d[NO].set[20 + OLID].local_set.data[3] = temp.data[3] + quantity * price;
            neworder_d[NO].set[20 + OLID].local_set.data[4] = temp.data[4] + quantity;
            // printf("neworder_d[NO].set[20 + OLID].local_set.data[2] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[2]);
            // printf("neworder_d[NO].set[20 + OLID].local_set.data[3] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[3]);
            // printf("neworder_d[NO].set[20 + OLID].local_set.data[4] = %d\n", neworder_d[NO].set[5 + OLID].local_set.data[4]);
            int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

            atomicAdd(&flag[htblloc].W_CNT, 1);
            bool blocked = true;
            // printf("start %d:%d\n", 7, OLID);
            while (blocked)
            {
                if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
                {
                    // printf("lock %d:%d\n", 7, OLID);
                    atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                    atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                    blocked = false;
                    // printf("unlock %d:%d\n", 7, OLID);
                }
            }
            // printf("end %d:%d\n", 7, OLID);
        }
    }

    __device__ void op_n_7(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // insert orderline loop
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        int O_OL_CNT = neworder_d[NO].query.O_OL_CNT;
        if (OLID < O_OL_CNT)
        {
            // printf("op_n_d[%d] wID : %d\t thID : %d\t\n", 35 + OLID, wID, thID);

            int TID = neworder_d[NO].TID;
            int WID = neworder_d[NO].query.W_ID;
            int OOLID = neworder_d[NO].query.O_OL_ID;
            int quantity = neworder_d[NO].query.INFO[OLID].OL_QUANTITY;
            int OLIID = neworder_d[NO].query.INFO[OLID].OL_I_ID;
            int price = current_2D_d[TABLE_SIZE_1D - ITEM_SIZE + OLIID].data[2];
            // neworder_d[NO].set[35 + OLID].WID = WID;
            int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + WID * 450000 + OOLID + OLID;
            neworder_d[NO].set[35 + OLID].Loc = Loc;
            // printf("neworder_d[NO].set[35 + %d].Loc = %d\n", OLID, neworder_d[NO].set[35 + OLID].Loc);

            neworder_d[NO].set[35 + OLID].local_set.data[0] = neworder_d[NO].query.O_OL_ID + OLID;
            neworder_d[NO].set[35 + OLID].local_set.data[1] = neworder_d[NO].query.O_ID;
            neworder_d[NO].set[35 + OLID].local_set.data[2] = neworder_d[NO].query.W_ID;
            neworder_d[NO].set[35 + OLID].local_set.data[3] = 0;
            neworder_d[NO].set[35 + OLID].local_set.data[4] = neworder_d[NO].query.INFO[OLID].OL_I_ID;
            neworder_d[NO].set[35 + OLID].local_set.data[5] = neworder_d[NO].query.INFO[OLID].OL_SUPPLY_W_ID;
            neworder_d[NO].set[35 + OLID].local_set.data[6] = 0;
            neworder_d[NO].set[35 + OLID].local_set.data[7] = neworder_d[NO].query.INFO[OLID].OL_QUANTITY;
            neworder_d[NO].set[35 + OLID].local_set.data[8] = quantity * price; // neworder_d[NO].query.INFO[OLID].OL_QUANTITY * price;
            neworder_d[NO].set[35 + OLID].local_set.data[9] = 0;
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[0] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[0]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[1] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[1]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[2] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[2]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[3] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[3]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[4] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[4]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[5] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[5]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[6] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[6]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[7] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[7]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[8] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[8]);
            // printf("neworder_d[NO].set[35 + OLID].local_set.data[9] = %d\n", neworder_d[NO].set[35 + OLID].local_set.data[9]);
            int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

            atomicAdd(&flag[htblloc].W_CNT, 1);
            bool blocked = true;
            // printf("start %d:%d\n", 8, OLID);
            while (blocked)
            {
                if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
                {
                    // printf("lock %d:%d\n", 8, OLID);
                    atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                    atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                    blocked = false;
                    // printf("unlock %d:%d\n", 8, OLID);
                }
            }
            // printf("end %d:%d\n", 8, OLID);
        }
    }

    typedef void (*OP_PTR)(int, int *, int, GLOBAL *, Set_n *, FLAG *, FLAGCOMP *);
    __device__ OP_PTR op_d[8] = {op_n_0, op_n_1, op_n_2, op_n_3, op_n_4, op_n_5, op_n_6, op_n_7};
    int op_type[8] = {SELECT, SELECT, SELECT, INSERT, INSERT, SELECT, UPDATE, INSERT};

    __device__ void execute_n(int *config, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        // printf("exe n %d:%d\n", thID, wID);
        if (wID >= config[0] && wID < config[0] + config[1])
        {
            for (size_t i = thID - 32 * config[0]; i < config[3]; i += 32 * config[1])
            {
                // printf("op_n_d[0] wID : %d\t thID : %d\t\n", wID, thID);
                op_d[0](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[4] && wID < config[4] + config[5])
        {
            for (size_t i = thID - 32 * config[4]; i < config[7]; i += 32 * config[5])
            {
                // printf("op_n_d[1] wID : %d\t thID : %d\t\n", wID, thID);
                op_d[1](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[8] && wID < config[8] + config[9])
        {
            for (size_t i = thID - 32 * config[8]; i < config[11]; i += 32 * config[9])
            {
                // printf("op_n_d[2] wID : %d\t thID : %d\t\n", wID, thID);
                op_d[2](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[12] && wID < config[12] + config[13])
        {
            for (size_t i = thID - 32 * config[12]; i < config[15]; i += 32 * config[13])
            {
                // printf("op_n_d[3] wID : %d\t thID : %d\t\n", wID, thID);
                op_d[3](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[16] && wID < config[16] + config[17])
        {
            for (size_t i = thID - 32 * config[16]; i < config[19]; i += 32 * config[17])
            {
                // printf("op_n_d[5] wID : %d\t thID : %d\t\n", wID, thID);
                // op_d[4](i, config, 0, current_2D_d, neworder_d, flag);
                op_d[4](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[20] && wID < config[20] + config[21])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[20]) / group_split.size();
            while (NO < config[23])
            {
                op_d[5](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[5](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag, flagcomp);
                NO += 32 / group_split.size() * config[21];
            }
            
        }
        else if (wID >= config[24] && wID < config[24] + config[25])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[24]) / group_split.size();
            while (NO < config[27])
            {
                // if (group_split.thread_rank() == 0)
                // {
                //     printf("NO.%d exec\n", NO);
                // }
                op_d[6](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[6](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag, flagcomp);
                NO += 32 / group_split.size() * config[25];
            }
        }
        else if (wID >= config[28] && wID < config[28] + config[29])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[28]) / group_split.size();
            while (NO < config[31])
            {
                // if (group_split.thread_rank() == 0)
                // {
                //     printf("NO.%d exec\n", NO);
                // }
                op_d[7](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag, flagcomp);
                // op_d[7](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag, flagcomp);
                NO += 32 / group_split.size() * config[29];
            }
        }
    }

    __device__ void wcheck(int NO, int OID, FLAG *flag, FLAGCOMP *flagcomp, Set_n *neworder_d, int OLID)
    {
        if (OLID < neworder_d[NO].query.O_OL_CNT)
        {
            int Loc = neworder_d[NO].set[OID].Loc;
            int htblloc = Loc; // findhash(Loc, OP_SIZE, flagcomp);

            int TID = neworder_d[NO].TID;
            // printf("loc = %d\n", loc);
            if (flag[htblloc].W_CNT > 1)
            {
                int minTID = 1 << 31 - 1;
                if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
                {
                    for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                    {
                        int tmp = flagcomp[htblloc].TID_LIST_W[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }
                else
                {
                    for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                    {
                        int tmp = flag[htblloc].TID_LIST_W[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }

                if (minTID < TID)
                {
                    atomicAdd(&neworder_d[NO].waw, 1);
                    // printf("waw minTID : %d; TID : %d; OID : %d; loc : %d\n", minTID, TID, OID, loc);
                }
            }
            if (flag[htblloc].R_CNT > 1)
            {
                int minTID = 1 << 31 - 1;
                if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
                {
                    for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                    {
                        int tmp = flagcomp[htblloc].TID_LIST_R[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }
                else
                {
                    for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                    {
                        int tmp = flag[htblloc].TID_LIST_R[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }

                if (minTID < TID)
                {
                    atomicAdd(&neworder_d[NO].war, 1);
                    // printf("war minTID : %d; TID : %d; NO : %d; OID : %d; loc : %d\n", minTID, TID, NO, OID, loc);
                }
            }
            // printf("end wcheck\n");
        }
    }

    __device__ void rcheck(int NO, int OID, FLAG *flag, FLAGCOMP *flagcomp, Set_n *neworder_d, int OLID)
    {
        if (OLID < neworder_d[NO].query.O_OL_CNT)
        {
            int Loc = neworder_d[NO].set[OID].Loc;
            int htblloc = Loc; // findhash(Loc, OP_SIZE, flagcomp);
            int TID = neworder_d[NO].TID;
            // printf("loc = %d\n", loc);
            if (flag[htblloc].W_CNT > 1)
            {
                int minTID = 1 << 31 - 1;
                if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
                {
                    for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                    {
                        int tmp = flagcomp[htblloc].TID_LIST_W[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }
                else
                {
                    for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                    {
                        int tmp = flag[htblloc].TID_LIST_W[i];
                        minTID = minTID > tmp ? tmp : minTID;
                    }
                }

                if (minTID < TID)
                {
                    atomicAdd(&neworder_d[NO].raw, 1);
                    // printf("raw minTID : %d; TID : %d; NO : %d; OID : %d; loc : %d\n", minTID, TID, NO, OID, loc);
                }
            }
            // printf("end rcheck\n");
        }
    }

    __device__ void check_n(int *config, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        // printf("check_n %d : %d\n", wID, thID);
        if (wID >= config[0] && wID < config[0] + config[1])
        {
            for (size_t i = thID - 32 * config[0] + config[2]; i < config[2] + config[3]; i += 32 * config[1])
            {
                // printf("check n[0] wID : %d\t thID : %d\t\n", wID, thID);

                rcheck(i, 0, flag, flagcomp, neworder_d, 0);
            }
        }
        else if (wID >= config[4] && wID < config[4] + config[5])
        {
            for (size_t i = thID - 32 * config[4] + config[6]; i < config[6] + config[7]; i += 32 * config[5])
            {
                // printf("check n[1] wID : %d\t thID : %d\t\n", wID, thID);

                rcheck(i, 1, flag, flagcomp, neworder_d, 0);
            }
        }
        else if (wID >= config[8] && wID < config[8] + config[9])
        {
            for (size_t i = thID - 32 * config[8] + config[10]; i < config[10] + config[11]; i += 32 * config[9])
            {
                // printf("check n[2] wID : %d\t thID : %d\t\n", wID, thID);

                rcheck(i, 2, flag, flagcomp, neworder_d, 0);
            }
        }
        else if (wID >= config[12] && wID < config[12] + config[13])
        {
            for (size_t i = thID - 32 * config[12] + config[14]; i < config[14] + config[15]; i += 32 * config[13])
            {
                // printf("check n[3] wID : %d\t thID : %d\t\n", wID, thID);

                wcheck(i, 3, flag, flagcomp, neworder_d, 0);
            }
        }
        else if (wID >= config[16] && wID < config[16] + config[17])
        {
            for (size_t i = thID - 32 * config[16] + config[18]; i < config[18] + config[19]; i += 32 * config[17])
            {
                // printf("check n[4] wID : %d\t thID : %d\t\n", wID, thID);

                // wcheck(i, 4, flag, neworder_d, 0);
                wcheck(i, 4, flag, flagcomp, neworder_d, 0);
            }
        }
        else if (wID >= config[20] && wID < config[20] + config[21])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[20] + config[22]) / group_split.size();
            while (NO < config[23])
            {
                rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 0, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 0);
                rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 1, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 1);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 2, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 2);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 3, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 3);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 4, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 4);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 5, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 5);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 6, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 6);
                // rcheck(NO, 5 + group_split.thread_rank() + group_split.size() * 7, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 7);
                NO += 32 / group_split.size() * config[21];
            }
        }
        else if (wID >= config[24] && wID < config[24] + config[25])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[24] + config[26]) / group_split.size();
            while (NO < config[27])
            {
                wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 0, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 0);
                wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 1, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 1);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 2, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 2);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 3, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 3);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 4, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 4);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 5, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 5);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 6, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 6);
                // wcheck(NO, 20 + group_split.thread_rank() + group_split.size() * 7, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 7);
                NO += 32 / group_split.size() * config[25];
            }
        }
        else if (wID >= config[28] && wID < config[28] + config[29])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[28] + config[30]) / group_split.size();
            while (NO < config[31])
            {
                wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 0, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 0);
                wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 1, flag, flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 1);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 2, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 2);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 3, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 3);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 4, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 4);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 5, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 5);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 6, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 6);
                // wcheck(NO, 35 + group_split.thread_rank() + group_split.size() * 7, flag,flagcomp, neworder_d, group_split.thread_rank() + group_split.size() * 7);
                NO += 32 / group_split.size() * config[29];
            }
        }
    }

    __device__ void op_n_wb_0(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int WID = neworder_d[NO].set[3].WID;
        int Loc = neworder_d[NO].set[3].Loc;
        int state = neworder_d[NO].waw || (neworder_d[NO].war && neworder_d[NO].raw);
        if (state == 0)
        {
            GLOBAL temp = neworder_d[NO].set[3].local_set;
            current_2D_d[Loc].data[0] = temp.data[0];
            current_2D_d[Loc].data[1] = temp.data[1];
            current_2D_d[Loc].data[2] = temp.data[2];
        }
    }

    __device__ void op_n_wb_1(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int WID = neworder_d[NO].set[5].WID;
        int Loc = neworder_d[NO].set[5 + OLID].Loc;
        int state = neworder_d[NO].waw || (neworder_d[NO].war && neworder_d[NO].raw);
        int OOLCNT = neworder_d[NO].query.O_OL_CNT;
        if (OLID < OOLCNT && state == 0)
        {
            GLOBAL temp = neworder_d[NO].set[5 + OLID].local_set;
            current_2D_d[Loc].data[2] = temp.data[2];
            current_2D_d[Loc].data[3] = temp.data[3];
            current_2D_d[Loc].data[4] = temp.data[4];
        }
    }

    __device__ void op_n_wb_2(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int WID = neworder_d[NO].set[20].WID;
        int Loc = neworder_d[NO].set[20 + OLID].Loc;
        int state = neworder_d[NO].waw || (neworder_d[NO].war && neworder_d[NO].raw);
        int OOLCNT = neworder_d[NO].query.O_OL_CNT;
        if (OLID < OOLCNT && state == 0)
        {
            GLOBAL temp = neworder_d[NO].set[20 + OLID].local_set;
            current_2D_d[Loc].data[2] = temp.data[2];
            current_2D_d[Loc].data[3] = temp.data[3];
            current_2D_d[Loc].data[4] = temp.data[4];
        }
    }

    __device__ void op_n_wb_3(int NO, int *config, int OLID, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int WID = neworder_d[NO].set[35].WID;
        int Loc = neworder_d[NO].set[35 + OLID].Loc;
        int state = neworder_d[NO].waw || (neworder_d[NO].war && neworder_d[NO].raw);
        int OOLCNT = neworder_d[NO].query.O_OL_CNT;
        if (OLID < OOLCNT && state == 0)
        {
            GLOBAL temp = neworder_d[NO].set[35 + OLID].local_set;
            current_2D_d[Loc].data[0] = temp.data[0];
            current_2D_d[Loc].data[1] = temp.data[1];
            current_2D_d[Loc].data[2] = temp.data[2];
            current_2D_d[Loc].data[3] = temp.data[3];
            current_2D_d[Loc].data[4] = temp.data[4];
            current_2D_d[Loc].data[5] = temp.data[5];
            current_2D_d[Loc].data[6] = temp.data[6];
            current_2D_d[Loc].data[7] = temp.data[7];
            current_2D_d[Loc].data[8] = temp.data[8];
            current_2D_d[Loc].data[9] = temp.data[9];
        }
    }

    // typedef void (*OP_PTR_WB)(int, int *, int, GLOBAL *, Set_n *, FLAG *, FLAGCOMP *);
    __device__ OP_PTR op_wb_d[4] = {op_n_wb_0, op_n_wb_1, op_n_wb_2, op_n_wb_3};

    __device__ void write_n(int *config, GLOBAL *current_2D_d, Set_n *neworder_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        if (wID >= config[36] && wID < config[36] + config[37])
        {
            for (size_t i = thID - 32 * config[36]; i < config[39]; i += 32 * config[37])
            {
                // printf("op_wb_d[0] wID : %d\t thID : %d\t\n", wID, thID);

                op_wb_d[0](i, config, 0, current_2D_d, neworder_d, flag, flagcomp);
            }
        }
        else if (wID >= config[40] && wID < config[40] + config[41])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[40]) / group_split.size();
            while (NO < config[43])
            {
                op_wb_d[1](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[1](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag,flagcomp);
                NO += 32 / group_split.size() * config[41];
            }
        }
        else if (wID >= config[44] && wID < config[44] + config[45])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[44]) / group_split.size();
            while (NO < config[47])
            {
                op_wb_d[2](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[2](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag,flagcomp);
                NO += 32 / group_split.size() * config[45];
            }
        }
        else if (wID >= config[48] && wID < config[48] + config[49])
        {
            thread_block group = this_thread_block();
            thread_block_tile<CG_SIZE_CONST> group_split = tiled_partition<CG_SIZE_CONST>(group);
            int NO = (thID - 32 * config[48]) / group_split.size();
            while (NO < config[51])
            {
                op_wb_d[3](NO, config, group_split.thread_rank(), current_2D_d, neworder_d, flag, flagcomp);
                op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 1, current_2D_d, neworder_d, flag, flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 2, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 3, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 4, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 5, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 6, current_2D_d, neworder_d, flag,flagcomp);
                // op_wb_d[3](NO, config, group_split.thread_rank() + group_split.size() * 7, current_2D_d, neworder_d, flag,flagcomp);
                NO += 32 / group_split.size() * config[49];
            }
        }
    }
};

namespace PAYMENT_QUERY
{
    int p_op_size = 7;

    __device__ void op_p_0(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // update warehouse
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[0] wID : %d\t thID : %d\t\n", wID, thID);
        int TID = payment_d[NO].TID;
        int WID = payment_d[NO].query.W_ID;
        int YTD = payment_d[NO].query.H_AMOUNT;
        // payment_d[NO].set[0].WID = WID;
        int Loc = WID;
        payment_d[NO].set[0].Loc = Loc;

        payment_d[NO].set[0].local_set.data[2] = current_2D_d[Loc].data[2] + YTD;

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);
        atomicAdd(&flagcomp[htblloc].W_CNT, 1);

        bool blocked = true;
        // printf("start %d:0\n", NO);
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                // printf("lock %d:0\n", NO);
                atomicMin(&flagcomp[htblloc].TID_LIST_W[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
                // printf("unlock %d:0\n", NO);
            }
        }
        // printf("end %d:0\n", NO);
    }

    __device__ void op_p_1(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from warehouse
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[1] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int WID = payment_d[NO].query.W_ID;
        int Loc = WID;
        payment_d[NO].set[1].Loc = Loc;
        GLOBAL temp = current_2D_d[Loc];

        payment_d[NO].set[1].local_set.data[3] = temp.data[3];
        payment_d[NO].set[1].local_set.data[4] = temp.data[4];
        payment_d[NO].set[1].local_set.data[5] = temp.data[5];
        payment_d[NO].set[1].local_set.data[6] = temp.data[6];
        payment_d[NO].set[1].local_set.data[7] = temp.data[7];
        payment_d[NO].set[1].local_set.data[8] = temp.data[8];

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);
        atomicAdd(&flag[htblloc].R_CNT, 1);
        atomicAdd(&flagcomp[htblloc].R_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                atomicMin(&flagcomp[htblloc].TID_LIST_R[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_R[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    __device__ void op_p_2(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // update district
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[2] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int WID = payment_d[NO].query.W_ID;
        int DID = payment_d[NO].query.D_ID;
        int YTD = payment_d[NO].query.H_AMOUNT;
        // payment_d[NO].set[2].WID = WID;
        int Loc = WAREHOUSE_SIZE + WID * 10 + DID;
        payment_d[NO].set[2].Loc = Loc;
        GLOBAL temp = current_2D_d[Loc];

        payment_d[NO].set[2].local_set.data[3] = temp.data[3] + YTD;

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);
        atomicAdd(&flagcomp[htblloc].W_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                atomicMin(&flagcomp[htblloc].TID_LIST_W[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    __device__ void op_p_3(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from district
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[3] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int WID = payment_d[NO].query.W_ID;
        int DID = payment_d[NO].query.D_ID;
        // payment_d[NO].set[3].WID = WID;
        int Loc = WAREHOUSE_SIZE + WID * 10 + DID;
        payment_d[NO].set[3].Loc = Loc;
        GLOBAL temp = current_2D_d[Loc];

        payment_d[NO].set[3].local_set.data[5] = temp.data[5];
        payment_d[NO].set[3].local_set.data[6] = temp.data[6];
        payment_d[NO].set[3].local_set.data[7] = temp.data[7];
        payment_d[NO].set[3].local_set.data[8] = temp.data[8];
        payment_d[NO].set[3].local_set.data[9] = temp.data[9];
        payment_d[NO].set[3].local_set.data[10] = temp.data[10];

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);
        atomicAdd(&flagcomp[htblloc].W_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1, 0) == 1)
            {
                atomicMin(&flagcomp[htblloc].TID_LIST_W[TID % COMPETITION_HASH_SIZE], TID);
                atomicExch(&flagcomp[htblloc].lock_W[TID % COMPETITION_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    __device__ void op_p_4(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // select from customer branch = 2
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[4] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int CWID = payment_d[NO].query.C_W_ID;
        int CDID = payment_d[NO].query.C_D_ID;
        if (payment_d[NO].query.isName)
        {
            int C_LAST = payment_d[NO].query.C_LAST;

            for (size_t i = 0; i < 3000; i++)
            {
                if (current_2D_d[WAREHOUSE_SIZE + DISTRICT_SIZE + CWID * 30000 + CDID * 3000 + i].data[13] == C_LAST)
                {
                    payment_d[NO].query.C_ID = current_2D_d[WAREHOUSE_SIZE + DISTRICT_SIZE + CWID * 30000 + CDID * 3000 + i].data[0];
                    break;
                }
            }
        }
        int CID = payment_d[NO].query.C_ID;
        GLOBAL temp = current_2D_d[WAREHOUSE_SIZE + DISTRICT_SIZE + CWID * 30000 + CDID * 3000 + CID];
        payment_d[NO].set[4].local_set.data[3] = temp.data[3];
        payment_d[NO].set[4].local_set.data[4] = temp.data[4];
        payment_d[NO].set[4].local_set.data[5] = temp.data[5];
        payment_d[NO].set[4].local_set.data[11] = temp.data[11];
        payment_d[NO].set[4].local_set.data[12] = temp.data[12];
        payment_d[NO].set[4].local_set.data[13] = temp.data[13];
        payment_d[NO].set[4].local_set.data[14] = temp.data[14];
        payment_d[NO].set[4].local_set.data[15] = temp.data[15];
        payment_d[NO].set[4].local_set.data[16] = temp.data[16];
        payment_d[NO].set[4].local_set.data[17] = temp.data[17];
        payment_d[NO].set[4].local_set.data[18] = temp.data[18];
        payment_d[NO].set[4].local_set.data[19] = temp.data[19];
        payment_d[NO].set[4].local_set.data[20] = temp.data[20];

        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CWID * 30000 + CDID * 3000 + CID;
        payment_d[NO].set[4].Loc = Loc;

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {

                atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);

                atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    __device__ void op_p_5(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // update customer  branch = 2
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[5] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int CID = payment_d[NO].query.C_ID;
        int CWID = payment_d[NO].query.C_W_ID;
        int CDID = payment_d[NO].query.C_D_ID;
        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CWID * 30000 + CDID * 3000 + CID;
        payment_d[NO].set[5].Loc = Loc;
        GLOBAL temp = current_2D_d[Loc];

        int C_CREDIT = temp.data[10];

        if (C_CREDIT) // good credit
        {
            // payment_d[NO].set[5].WID = CWID;

            payment_d[NO].set[5].local_set.data[5] = temp.data[5] + payment_d[NO].query.H_AMOUNT;
        }
        else
        {
            // payment_d[NO].set[5].WID = CWID;

            payment_d[NO].set[5].local_set.data[9] = temp.data[9] + payment_d[NO].query.H_AMOUNT;
        }

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);

        atomicAdd(&flag[htblloc].W_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {
                atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    __device__ void op_p_6(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    { // insert history
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_p_d[6] wID : %d\t thID : %d\t\n", wID, thID);

        int TID = payment_d[NO].TID;
        int WID = payment_d[NO].query.W_ID;
        int HID = payment_d[NO].query.H_ID;
        // payment_d[NO].set[6].WID = WID;
        int Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + WID * 30000 + HID;
        payment_d[NO].set[6].Loc = Loc;
        payment_d[NO].set[6].local_set.data[0] = payment_d[NO].query.C_ID;
        payment_d[NO].set[6].local_set.data[1] = payment_d[NO].query.C_D_ID;
        payment_d[NO].set[6].local_set.data[2] = payment_d[NO].query.C_W_ID;
        payment_d[NO].set[6].local_set.data[3] = payment_d[NO].query.D_ID;
        payment_d[NO].set[6].local_set.data[4] = WID;

        int htblloc = Loc; // hash(Loc, OP_SIZE, flagcomp);
        atomicAdd(&flag[htblloc].W_CNT, 1);

        bool blocked = true;
        while (blocked)
        {
            if (atomicCAS(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1, 0) == 1)
            {
                atomicMin(&flag[htblloc].TID_LIST_W[TID % NORMAL_HASH_SIZE], TID);
                atomicExch(&flag[htblloc].lock_W[TID % NORMAL_HASH_SIZE], 1);
                blocked = false;
            }
        }
    }

    typedef void (*OP_PTR)(int, int *, GLOBAL *, Set_p *, FLAG *, FLAGCOMP *);
    __device__ OP_PTR op_d[7] = {op_p_0, op_p_1, op_p_2, op_p_3, op_p_4, op_p_5, op_p_6};
    int op_type[7] = {UPDATE, SELECT, UPDATE, SELECT, SELECT, UPDATE, INSERT};

    __device__ void execute_p(int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        if (wID >= config[0] && wID < config[0] + config[1])
        {
            for (size_t i = thID - 32 * config[0]; i < config[3]; i += 32 * config[1])
            {
                // printf("op_p_d[0] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[0](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[4] && wID < config[4] + config[5])
        {
            for (size_t i = thID - 32 * config[4]; i < config[7]; i += 32 * config[5])
            {
                // printf("op_p_d[1] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[1](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[8] && wID < config[8] + config[9])
        {
            for (size_t i = thID - 32 * config[8]; i < config[11]; i += 32 * config[9])
            {
                // printf("op_p_d[2] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[2](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[12] && wID < config[12] + config[13])
        {
            for (size_t i = thID - 32 * config[12]; i < config[15]; i += 32 * config[13])
            {
                // printf("op_p_d[3] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[3](i, config, current_2D_d, payment_d, flag, flagcomp);
                op_d[4](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[16] && wID < config[16] + config[17])
        {
            for (size_t i = thID - 32 * config[16]; i < config[19]; i += 32 * config[17])
            {
                // printf("op_p_d[5] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[5](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[20] && wID < config[20] + config[21])
        {
            for (size_t i = thID - 32 * config[20]; i < config[23]; i += 32 * config[21])
            {
                // printf("op_p_d[6] wID : %d\t thID : %d\t\n", wID, thID);

                op_d[6](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
    }

    __device__ void wcheck(FLAG *flag, FLAGCOMP *flagcomp, Set_p *payment_d, int NO, int OID)
    {
        int Loc = payment_d[NO].set[OID].Loc;
        int htblloc = Loc; // findhash(Loc, OP_SIZE, flagcomp);
        int TID = payment_d[NO].TID;
        // printf("wcheck loc = %d\n", loc);
        if (flag[htblloc].W_CNT > 1) // && loc > WAREHOUSE_SIZE)
        {
            int minTID = 1 << 31 - 1;
            if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
            {
                for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                {
                    int tmp = flagcomp[htblloc].TID_LIST_W[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }
            else
            {
                for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                {
                    int tmp = flag[htblloc].TID_LIST_W[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }

            if (minTID < TID)
            {
                atomicAdd(&payment_d[NO].waw, 1);
                // printf("waw minTID : %d; TID : %d; NO : %d; OID : %d; loc : %d\n", minTID, TID, NO, OID, loc);
            }
        }
        if (flag[htblloc].R_CNT > 1) // && loc > WAREHOUSE_SIZE)
        {
            int minTID = 1 << 31 - 1;
            if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
            {
                for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                {
                    int tmp = flagcomp[htblloc].TID_LIST_R[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }
            else
            {
                for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                {
                    int tmp = flag[htblloc].TID_LIST_R[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }

            if (minTID < TID)
            {
                atomicAdd(&payment_d[NO].war, 1);
                // printf("war minTID : %d; TID : %d; NO : %d; OID : %d; loc : %d\n", minTID, TID, NO, OID, loc);
            }
        }
        // printf("end wcheck\n");
    }

    __device__ void rcheck(FLAG *flag, FLAGCOMP *flagcomp, Set_p *payment_d, int NO, int OID)
    {
        int Loc = payment_d[NO].set[OID].Loc;
        int htblloc = Loc; // findhash(Loc, OP_SIZE, flagcomp);
        int TID = payment_d[NO].TID;
        // printf("rcheck loc = %d\n", loc);
        if (flag[htblloc].W_CNT > 1) // && loc > WAREHOUSE_SIZE)
        {
            int minTID = 1 << 31 - 1;
            if (Loc < WAREHOUSE_SIZE + DISTRICT_SIZE)
            {
                for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
                {
                    int tmp = flagcomp[htblloc].TID_LIST_W[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }
            else
            {
                for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
                {
                    int tmp = flag[htblloc].TID_LIST_W[i];
                    minTID = minTID > tmp ? tmp : minTID;
                }
            }
            if (minTID < TID)
            {
                atomicAdd(&payment_d[NO].raw, 1);
                // printf("raw minTID : %d; TID : %d; NO : %d; OID : %d; loc : %d\n", minTID, TID, NO, OID, loc);
            }
        }
        // printf("end rcheck\n");
    }

    __device__ void check_p(int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        // printf("check p %d:%d\n", wID, thID);
        if (wID >= config[0] && wID < config[0] + config[1])
        {
            for (size_t i = thID - 32 * config[0] + config[2]; i < config[2] + config[3]; i += 32 * config[1])
            {
                // printf("check p[0] wID : %d\t thID : %d\t\n", wID, thID);

                wcheck(flag, flagcomp, payment_d, i, 0);
            }
        }
        else if (wID >= config[4] && wID < config[4] + config[5])
        {
            for (size_t i = thID - 32 * config[4] + config[6]; i < config[6] + config[7]; i += 32 * config[5])
            {
                // printf("check p[1] wID : %d\t thID : %d\t\n", wID, thID);

                rcheck(flag, flagcomp, payment_d, i, 1);
            }
        }
        else if (wID >= config[8] && wID < config[8] + config[9])
        {
            for (size_t i = thID - 32 * config[8] + config[10]; i < config[10] + config[11]; i += 32 * config[9])
            {
                // printf("check p[2] wID : %d\t thID : %d\t\n", wID, thID);

                wcheck(flag, flagcomp, payment_d, i, 2);
            }
        }
        else if (wID >= config[12] && wID < config[12] + config[13])
        {
            for (size_t i = thID - 32 * config[12] + config[14]; i < config[14] + config[15]; i += 32 * config[13])
            {
                // printf("check p[3] wID : %d\t thID : %d\t\n", wID, thID);

                rcheck(flag, flagcomp, payment_d, i, 3);
                rcheck(flag, flagcomp, payment_d, i, 4);
            }
        }
        else if (wID >= config[16] && wID < config[16] + config[17])
        {
            for (size_t i = thID - 32 * config[16] + config[18]; i < config[18] + config[19]; i += 32 * config[17])
            {
                // printf("check p[5] wID : %d\t thID : %d\t\n", wID, thID);

                wcheck(flag, flagcomp, payment_d, i, 5);
            }
        }
        else if (wID >= config[20] && wID < config[20] + config[21])
        {
            for (size_t i = thID - 32 * config[20] + config[22]; i < config[22] + config[23]; i += 32 * config[21])
            {
                // printf("check p[6] wID : %d\t thID : %d\t\n", wID, thID);

                wcheck(flag, flagcomp, payment_d, i, 6);
            }
        }
    }

    __device__ void op_p_wb_0(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_wb_d[0] wID : %d\t thID : %d\t\n", wID, thID);
        // printf("No.%d waw : %d\n", NO, payment_d[NO].waw);
        int Loc = payment_d[NO].set[0].Loc;
        int state = payment_d[NO].waw || (payment_d[NO].war && payment_d[NO].raw);
        if (!state)
        {
            GLOBAL temp = payment_d[NO].set[0].local_set;
            current_2D_d[Loc].data[2] = temp.data[2];
        }
    }

    __device__ void op_p_wb_1(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_wb_d[1] wID : %d\t thID : %d\t\n", wID, thID);

        int Loc = payment_d[NO].set[2].Loc;
        int state = payment_d[NO].waw || (payment_d[NO].war && payment_d[NO].raw);
        if (!state)
        {
            GLOBAL temp = payment_d[NO].set[2].local_set;
            current_2D_d[Loc].data[2] = temp.data[2];
        }
    }

    __device__ void op_p_wb_2(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_wb_d[2] wID : %d\t thID : %d\t\n", wID, thID);

        int Loc = payment_d[NO].set[2].Loc;
        int state = payment_d[NO].waw || (payment_d[NO].war && payment_d[NO].raw);
        if (!state)
        {
            GLOBAL temp = payment_d[NO].set[5].local_set;
            current_2D_d[Loc].data[5] = temp.data[5];
            current_2D_d[Loc].data[9] = temp.data[9];
        }
    }

    __device__ void op_p_wb_3(int NO, int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        // int thID = threadIdx.x + blockDim.x * blockIdx.x;
        // int wID = thID / 32;
        // printf("op_wb_d[3] wID : %d\t thID : %d\t\n", wID, thID);

        int Loc = payment_d[NO].set[2].Loc;
        int state = payment_d[NO].waw || (payment_d[NO].war && payment_d[NO].raw);
        if (!state)
        {
            GLOBAL temp = payment_d[NO].set[6].local_set;
            current_2D_d[Loc].data[0] = temp.data[0];
            current_2D_d[Loc].data[1] = temp.data[1];
            current_2D_d[Loc].data[2] = temp.data[2];
            current_2D_d[Loc].data[3] = temp.data[3];
            current_2D_d[Loc].data[4] = temp.data[4];
        }
    }

    __device__ OP_PTR op_wb_d[4] = {op_p_wb_0, op_p_wb_1, op_p_wb_2, op_p_wb_3};

    __device__ void write_p(int *config, GLOBAL *current_2D_d, Set_p *payment_d, FLAG *flag, FLAGCOMP *flagcomp)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int wID = thID / 32;
        if (wID >= config[24] && wID < config[24] + config[25])
        {
            for (size_t i = thID - 32 * config[24] + config[26]; i < config[26] + config[27]; i += 32 * config[25])
            {
                // printf("op_wb_d[0] wID : %d\t thID : %d\t\n", wID, thID);
                op_wb_d[0](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[28] && wID < config[28] + config[29])
        {
            for (size_t i = thID - 32 * config[28] + config[30]; i < config[30] + config[31]; i += 32 * config[29])
            {
                // printf("op_wb_d[1] wID : %d\t thID : %d\t\n", wID, thID);
                op_wb_d[1](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[32] && wID < config[32] + config[33])
        {
            for (size_t i = thID - 32 * config[32] + config[34]; i < config[34] + config[35]; i += 32 * config[33])
            {
                // printf("op_wb_d[2] wID : %d\t thID : %d\t\n", wID, thID);
                op_wb_d[2](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
        else if (wID >= config[36] && wID < config[36] + config[37])
        {
            for (size_t i = thID - 32 * config[36] + config[38]; i < config[38] + config[39]; i += 32 * config[37])
            {
                // printf("op_wb_d[3] wID : %d\t thID : %d\t\n", wID, thID);
                op_wb_d[3](i, config, current_2D_d, payment_d, flag, flagcomp);
            }
        }
    }

}