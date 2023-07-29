#pragma once
#include "Predefine.h"
struct YCSB_TABLE
{
    unsigned int Y_0;
    unsigned int Y_1;
    unsigned int Y_2;
    unsigned int Y_3;
    unsigned int Y_4;
    unsigned int Y_5;
    unsigned int Y_6;
    unsigned int Y_7;
    unsigned int Y_8;
    unsigned int Y_9;
};
struct YCSB_SNAPSHOT
{
    unsigned int data[YCSB_SIZE * YCSB_COLUMN];
};
struct YCSB_INDEX
{
    /* data */
};
struct YCSB_LOG
{
    unsigned int YCSB_R[YCSB_SIZE];
    unsigned int YCSB_W[YCSB_SIZE];
    unsigned int YTD[YCSB_A_SIZE];
    unsigned int TMP_YTD[YCSB_A_SIZE];
};
struct ycsbAQuery
{
    unsigned int LOC_R[YCSB_READ_SIZE];
    unsigned int LOC_W[YCSB_WRITE_SIZE];
};
struct YCSB_A_QUERY
{
    unsigned int LOC_R[YCSB_READ_SIZE * YCSB_A_SIZE];
    unsigned int LOC_W[YCSB_WRITE_SIZE * YCSB_A_SIZE];
    unsigned int TID[YCSB_A_SIZE];
};
struct YCSB_A_SET
{
    unsigned int set_Loc[YCSB_OP_SIZE * YCSB_A_SIZE];                     // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[YCSB_COLUMN * YCSB_OP_SIZE * YCSB_A_SIZE]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    unsigned int war[YCSB_OP_SIZE * YCSB_A_SIZE];
    unsigned int raw[YCSB_OP_SIZE * YCSB_A_SIZE];
    unsigned int waw[YCSB_OP_SIZE * YCSB_A_SIZE];
    unsigned int COMMIT_AND_ABORT[YCSB_A_SIZE];
    unsigned int COMMIT;
};
