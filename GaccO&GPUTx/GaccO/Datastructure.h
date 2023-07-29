#pragma once
#include "Predefine.h"
struct WAREHOUSE /* 9 */
{
    unsigned int W_ID;       // 0
    unsigned int W_TAX;      // 1
    unsigned int W_YTD;      // 2
    unsigned int W_NAME;     // 3
    unsigned int W_STREET_1; // 4
    unsigned int W_STREET_2; // 5
    unsigned int W_CITY;     // 6
    unsigned int W_STATE;    // 7
    unsigned int W_ZIP;      // 8
};
struct DISTRICT /* 11 */
{
    unsigned int D_ID;        // 0
    unsigned int D_W_ID;      // 1
    unsigned int D_TAX;       // 2
    unsigned int D_YTD;       // 3
    unsigned int D_NEXT_O_ID; // 4
    unsigned int D_NAME;      // 5
    unsigned int D_STREET_1;  // 6
    unsigned int D_STREET_2;  // 7
    unsigned int D_CITY;      // 8
    unsigned int D_STATE;     // 9
    unsigned int D_ZIP;       // 10
};
struct CUSTOMER /* 21 */
{
    unsigned int C_ID;           // 0
    unsigned int C_W_ID;         // 1
    unsigned int C_D_ID;         // 2
    unsigned int C_CREDIT_LIM;   // 3
    unsigned int C_DISCOUNT;     // 4
    unsigned int C_BALANCE;      // 5
    unsigned int C_YTD_PAYMENT;  // 6
    unsigned int C_PAYMENT_CNT;  // 7
    unsigned int C_DELIVERY_CNT; // 8
    unsigned int C_DATA;         // 9
    unsigned int C_CREDIT;       // 10
    unsigned int C_FIRST;        // 11
    unsigned int C_MIDDLE;       // 12
    unsigned int C_LAST;         // 13
    unsigned int C_STREET_1;     // 14
    unsigned int C_STREET_2;     // 15
    unsigned int C_CITY;         // 16
    unsigned int C_STATE;        // 17
    unsigned int C_ZIP;          // 18
    unsigned int C_PHONE;        // 19
    unsigned int C_SINCE;        // 20
};
struct HISTORY /* 5 */
{
    unsigned int H_C_ID;   // 0
    unsigned int H_C_D_ID; // 1
    unsigned int H_C_W_ID; // 2
    unsigned int H_D_ID;   // 3
    unsigned int H_W_ID;   // 4
};
struct NEWORDER /* 3 */
{
    unsigned int NO_O_ID; // 0
    unsigned int NO_D_ID; // 1
    unsigned int NO_W_ID; // 2
};
struct ORDER /* 8 */
{
    unsigned int O_ID;         // 0
    unsigned int O_D_ID;       // 1
    unsigned int O_W_ID;       // 2
    unsigned int O_C_ID;       // 3
    unsigned int O_ENTRY_D;    // 4
    unsigned int O_CARRIER_ID; // 5
    unsigned int O_OL_CNT;     // 6
    unsigned int O_ALL_LOCAL;  // 7
};
struct ORDERLINE /* 10 */
{
    unsigned int OL_O_ID;        // 0
    unsigned int OL_D_ID;        // 1
    unsigned int OL_W_ID;        // 2
    unsigned int OL_NUMBER;      // 3
    unsigned int OL_I_ID;        // 4
    unsigned int OL_SUPPLY_W_ID; // 5
    unsigned int OL_DELIVERY_D;  // 6
    unsigned int OL_QUANLITY;    // 7
    unsigned int OL_AMOUNT;      // 8
    unsigned int OL_DIST_INF;    // 9
};
struct STOCK /* 17 */
{
    unsigned int S_I_ID;       // 0
    unsigned int S_W_ID;       // 1
    unsigned int S_QUANTITY;   // 2
    unsigned int S_YTD;        // 3
    unsigned int S_ORDER_CNT;  // 4
    unsigned int S_REMOVE_CNT; // 5
    unsigned int S_DIST_01;    // 6
    unsigned int S_DIST_02;    // 7
    unsigned int S_DIST_03;    // 8
    unsigned int S_DIST_04;    // 9
    unsigned int S_DIST_05;    // 10
    unsigned int S_DIST_06;    // 11
    unsigned int S_DIST_07;    // 12
    unsigned int S_DIST_08;    // 13
    unsigned int S_DIST_09;    // 14
    unsigned int S_DIST_10;    // 15
    unsigned int S_DATA;       // 16
};
struct ITEM /* 5 */
{
    unsigned int I_ID;    // 0
    unsigned int I_IM_ID; // 1
    unsigned int I_PRICE; // 2
    unsigned int I_NAME;  // 3
    unsigned int I_DATA;  // 4
};
struct SNAPSHOT // 相比原始表多一位第0位, 用于READ/WRITE时写入空数据
{
    unsigned int warehouse_snapshot[WAREHOUSE_SIZE * 9];  // 9
    unsigned int district_snapshot[DISTRICT_SIZE * 11];   // 11
    unsigned int customer_snapshot[CUSTOMER_SIZE * 21];   // 21
    unsigned int history_snapshot[HISTORY_SIZE * 5];      // 5
    unsigned int neworder_snapshot[NEWORDER_SIZE * 3];    // 3
    unsigned int order_snapshot[ORDER_SIZE * 8];          // 8
    unsigned int orderline_snapshot[ORDERLINE_SIZE * 10]; // 10
    unsigned int stock_snapshot[STOCK_SIZE * 17];         // 17
    unsigned int item_snapshot[ITEM_SIZE * 5];            // 5
};
struct INDEX
{
    unsigned int customer_name_index[CUSTOMER_SIZE];
    unsigned int orderline_index[WAREHOUSE_SIZE];
    unsigned int *index; // defined, but didn't use
};
struct LOG
{
    unsigned int LOG_WAREHOUSE_R[WAREHOUSE_SIZE * 2];
    unsigned int LOG_WAREHOUSE_W[WAREHOUSE_SIZE * 2];
    unsigned int LOG_DISTRICT_R[DISTRICT_SIZE * 2];
    unsigned int LOG_DISTRICT_W[DISTRICT_SIZE * 2];
    unsigned int LOG_CUSTOMER_R[CUSTOMER_SIZE];
    unsigned int LOG_CUSTOMER_W[CUSTOMER_SIZE];
    unsigned int LOG_HISTORY_R[HISTORY_SIZE];
    unsigned int LOG_HISTORY_W[HISTORY_SIZE];
    unsigned int LOG_NEWORDER_R[NEWORDER_SIZE];
    unsigned int LOG_NEWORDER_W[NEWORDER_SIZE];
    unsigned int LOG_ORDER_R[ORDER_SIZE];
    unsigned int LOG_ORDER_W[ORDER_SIZE];
    unsigned int LOG_ORDERLINE_R[ORDERLINE_SIZE];
    unsigned int LOG_ORDERLINE_W[ORDERLINE_SIZE];
    unsigned int LOG_STOCK_R[STOCK_SIZE];
    unsigned int LOG_STOCK_W[STOCK_SIZE];
    unsigned int LOG_ITEM_R[ITEM_SIZE];
    unsigned int LOG_ITEM_W[ITEM_SIZE];
    unsigned int W_YTD[PAYMENT_CNT];
    unsigned int TMP_W_YTD[WAREHOUSE_SIZE];
    unsigned int D_YTD[PAYMENT_CNT];
    unsigned int TMP_D_YTD[DISTRICT_SIZE];
};
struct NeworderQuery
{
    unsigned int W_ID;
    unsigned int D_ID;
    unsigned int C_ID;
    unsigned int O_ID;
    unsigned int N_O_ID;
    unsigned int O_OL_CNT;
    unsigned int O_OL_ID;
    struct NewOrderQueryInfo
    {
        unsigned int OL_I_ID;
        unsigned int OL_SUPPLY_W_ID;
        unsigned int OL_QUANTITY;
    };
    NewOrderQueryInfo INFO[15];
};
struct PaymentQuery
{
    unsigned int W_ID;
    unsigned int D_ID;
    unsigned int C_ID;
    unsigned int C_LAST;
    unsigned int isName; // 0,id; 1,name
    unsigned int C_D_ID;
    unsigned int C_W_ID;
    unsigned int H_AMOUNT;
    unsigned int H_ID;
};
struct OrderstatusQuery
{
    unsigned int W_ID;
    unsigned int D_ID;
    unsigned int C_ID;
    unsigned int C_LAST;
    unsigned int O_ID;
    unsigned int OL_ID;
    unsigned int isName; // 0,id; 1,name
};
struct DeliveryQuery
{
    unsigned int O_ID[10];
};
struct StockLevelQuery
{
    unsigned int query_cnt;
    unsigned int W_ID;
    unsigned int D_ID;
};
struct NEWORDER_QUERY
{
    unsigned int W_ID[NEWORDER_CNT];
    unsigned int D_ID[NEWORDER_CNT];
    unsigned int C_ID[NEWORDER_CNT];
    unsigned int O_ID[NEWORDER_CNT];
    unsigned int N_O_ID[NEWORDER_CNT];
    unsigned int O_OL_CNT[NEWORDER_CNT];
    unsigned int O_OL_ID[NEWORDER_CNT];
    unsigned int OL_I_ID[15 * NEWORDER_CNT];
    unsigned int OL_SUPPLY_W_ID[15 * NEWORDER_CNT];
    unsigned int OL_QUANTITY[15 * NEWORDER_CNT];
    unsigned int TID[NEWORDER_CNT];
};
struct NEWORDER_SET
{
    // NeworderQuery query[NEWORDER_CNT];
    unsigned int set_Loc[NEWORDER_OP_CNT * NEWORDER_CNT];            // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[32 * NEWORDER_OP_CNT * NEWORDER_CNT]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    // unsigned int TID[NEWORDER_CNT];                                  // 12-BATCH_ID 20-TID
    unsigned int war[NEWORDER_OP_CNT * NEWORDER_CNT];
    unsigned int raw[NEWORDER_OP_CNT * NEWORDER_CNT];
    unsigned int waw[NEWORDER_OP_CNT * NEWORDER_CNT];
    unsigned int COMMIT_AND_ABORT[NEWORDER_CNT];
    unsigned int COMMIT;
};
struct PAYMENT_QUERY
{
    unsigned int W_ID[PAYMENT_CNT];
    unsigned int D_ID[PAYMENT_CNT];
    unsigned int C_ID[PAYMENT_CNT];
    unsigned int C_LAST[PAYMENT_CNT];
    unsigned int isName[PAYMENT_CNT]; // 0,id; 1,name
    unsigned int C_D_ID[PAYMENT_CNT];
    unsigned int C_W_ID[PAYMENT_CNT];
    unsigned int H_AMOUNT[PAYMENT_CNT];
    unsigned int H_ID[PAYMENT_CNT];
    unsigned int TID[PAYMENT_CNT];
};
struct PAYMENT_SET
{
    // PaymentQuery query[PAYMENT_CNT];
    unsigned int set_Loc[PAYMENT_OP_CNT * PAYMENT_CNT];            // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[32 * PAYMENT_OP_CNT * PAYMENT_CNT]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    // unsigned int TID[PAYMENT_CNT];                                 // 12-BATCH_ID 20-ID
    unsigned int war[PAYMENT_OP_CNT * PAYMENT_CNT];
    unsigned int raw[PAYMENT_OP_CNT * PAYMENT_CNT];
    unsigned int waw[PAYMENT_OP_CNT * PAYMENT_CNT];
    unsigned int COMMIT_AND_ABORT[PAYMENT_CNT];
    unsigned int COMMIT;
};
struct ORDERSTATUS_SET
{
    OrderstatusQuery query[ORDERSTATUS_CNT];
    unsigned int set_Loc[ORDERSTATUS_OP_CNT * ORDERSTATUS_CNT];            // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[32 * ORDERSTATUS_OP_CNT * ORDERSTATUS_CNT]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    unsigned int TID[ORDERSTATUS_CNT];                                     // 12-BATCH_ID 20-ID
    unsigned int war[ORDERSTATUS_OP_CNT * ORDERSTATUS_CNT];
    unsigned int raw[ORDERSTATUS_OP_CNT * ORDERSTATUS_CNT];
    unsigned int waw[ORDERSTATUS_OP_CNT * ORDERSTATUS_CNT];
    unsigned int COMMIT_AND_ABORT[ORDERSTATUS_CNT];
    unsigned int COMMIT;
};
struct DELIVERY_SET
{
    DeliveryQuery query[DELIVERY_CNT];
    unsigned int set_Loc[DELIVERY_OP_CNT * DELIVERY_CNT];            // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[32 * DELIVERY_OP_CNT * DELIVERY_CNT]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    unsigned int TID[DELIVERY_CNT];                                  // 12-BATCH_ID 20-ID
    unsigned int war[DELIVERY_OP_CNT * DELIVERY_CNT];
    unsigned int raw[DELIVERY_OP_CNT * DELIVERY_CNT];
    unsigned int waw[DELIVERY_OP_CNT * DELIVERY_CNT];
    unsigned int COMMIT_AND_ABORT[DELIVERY_CNT];
    unsigned int COMMIT;
};
struct STOCK_SET
{
    StockLevelQuery query[STOCK_CNT];
    unsigned int set_Loc[STOCK_OP_CNT * STOCK_CNT];            // set_Loc 是 SET 中的 Loc, 以 SoA 的形式实现
    unsigned int set_local_set[32 * STOCK_OP_CNT * STOCK_CNT]; // set_local_set 是 SET 中的 local_set, 以 SoA 的形式实现
    unsigned int TID[STOCK_CNT];                               // 12-BATCH_ID 20-ID
    unsigned int war[STOCK_OP_CNT * STOCK_CNT];
    unsigned int raw[STOCK_OP_CNT * STOCK_CNT];
    unsigned int waw[STOCK_OP_CNT * STOCK_CNT];
    unsigned int COMMIT_AND_ABORT[STOCK_CNT];
    unsigned int COMMIT;
};

struct WAREHOUSE_ACCESS
{
    unsigned int W_ID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    unsigned int TID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    // unsigned int LENGTH = NEWORDER_CNT + 2 * PAYMENT_CNT;
};
struct WAREHOUSE_AUXILIARY
{
    unsigned int W_ID[WAREHOUSE_SIZE];
    unsigned int HIST[WAREHOUSE_SIZE];
    unsigned int OFFSET[WAREHOUSE_SIZE];
    unsigned int LOCK[WAREHOUSE_SIZE];
};
struct DISTRICT_ACCESS
{
    unsigned int D_ID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    unsigned int TID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    // unsigned int LENGTH = NEWORDER_CNT + 2 * PAYMENT_CNT;
};
struct DISTRICT_AUXILIARY
{
    unsigned int D_ID[DISTRICT_SIZE];
    unsigned int HIST[DISTRICT_SIZE];
    unsigned int OFFSET[DISTRICT_SIZE];
    unsigned int LOCK[DISTRICT_SIZE];
};
struct CUSTOMER_ACCESS
{
    unsigned int C_ID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    unsigned int TID[NEWORDER_CNT + 2 * PAYMENT_CNT];
    // unsigned int LENGTH = NEWORDER_CNT + 2 * PAYMENT_CNT;
};
struct CUSTOMER_AUXILIARY
{
    unsigned int C_ID[CUSTOMER_SIZE];
    unsigned int HIST[CUSTOMER_SIZE];
    unsigned int OFFSET[CUSTOMER_SIZE];
    unsigned int LOCK[CUSTOMER_SIZE];
};
struct HISTORY_ACCESS
{
    unsigned int H_ID[PAYMENT_CNT];
    unsigned int TID[PAYMENT_CNT];
    // unsigned int LENGTH = PAYMENT_CNT;
};
struct HISTORY_AUXILIARY
{
    unsigned int H_ID[HISTORY_SIZE];
    unsigned int HIST[HISTORY_SIZE];
    unsigned int OFFSET[HISTORY_SIZE];
    unsigned int LOCK[HISTORY_SIZE];
};
struct NEWORDER_ACCESS
{
    unsigned int N_ID[NEWORDER_CNT];
    unsigned int TID[NEWORDER_CNT];
    // unsigned int LENGTH = NEWORDER_CNT;
};
struct NEWORDER_AUXILIARY
{
    unsigned int N_ID[NEWORDER_SIZE];
    unsigned int HIST[NEWORDER_SIZE];
    unsigned int OFFSET[NEWORDER_SIZE];
    unsigned int LOCK[NEWORDER_SIZE];
};
struct ORDER_ACCESS
{
    unsigned int O_ID[NEWORDER_CNT];
    unsigned int TID[NEWORDER_CNT];
    // unsigned int LENGTH = NEWORDER_CNT;
};
struct ORDER_AUXILIARY
{
    unsigned int O_ID[ORDER_SIZE];
    unsigned int HIST[ORDER_SIZE];
    unsigned int OFFSET[ORDER_SIZE];
    unsigned int LOCK[ORDER_SIZE];
};
struct ORDERLINE_ACCESS
{
    unsigned int OL_ID[NEWORDER_CNT * 15];
    unsigned int TID[NEWORDER_CNT * 15];
    // unsigned int LENGTH = NEWORDER_CNT * 15;
};
struct ORDERLINE_AUXILIARY
{
    unsigned int OL_ID[ORDERLINE_SIZE];
    unsigned int HIST[ORDERLINE_SIZE];
    unsigned int OFFSET[ORDERLINE_SIZE];
    unsigned int LOCK[ORDERLINE_SIZE];
};
struct STOCK_ACCESS
{
    unsigned int S_ID[NEWORDER_CNT * 15];
    unsigned int TID[NEWORDER_CNT * 15];
    // unsigned int LENGTH = NEWORDER_CNT * 15;
};
struct STOCK_AUXILIARY
{
    unsigned int S_ID[STOCK_SIZE];
    unsigned int HIST[STOCK_SIZE];
    unsigned int OFFSET[STOCK_SIZE];
    unsigned int LOCK[STOCK_SIZE];
};
struct ITEM_ACCESS
{
    unsigned int I_ID[NEWORDER_CNT * 15];
    unsigned int TID[NEWORDER_CNT * 15];
    // unsigned int LENGTH = NEWORDER_CNT * 15;
};
struct ITEM_AUXILIARY
{
    unsigned int I_ID[ITEM_SIZE];
    unsigned int HIST[ITEM_SIZE];
    unsigned int OFFSET[ITEM_SIZE];
    unsigned int LOCK[ITEM_SIZE];
};
struct NEWORDERQUERY_ACCESS
{
    WAREHOUSE_ACCESS warehouse_access;
    DISTRICT_ACCESS district_access;
    CUSTOMER_ACCESS customer_access;
    NEWORDER_ACCESS neworder_access;
    ORDER_ACCESS order_access;
    ORDERLINE_ACCESS orderline_access;
    STOCK_ACCESS stock_access;
    ITEM_ACCESS item_access;
};
struct PAYMENTQUERY_ACCESS
{
    WAREHOUSE_ACCESS warehouse_access;
    DISTRICT_ACCESS district_access;
    CUSTOMER_ACCESS customer_access;
    HISTORY_ACCESS history_access;
};
struct NEWORDERQUERY_AUXILIARY
{
    WAREHOUSE_AUXILIARY warehouse_auxiliary;
    DISTRICT_AUXILIARY distrct_auxiliary;
    CUSTOMER_AUXILIARY customer_auxiliary;
    NEWORDER_AUXILIARY neworder_auxiliary;
    ORDER_AUXILIARY order_auxiliary;
    ORDERLINE_AUXILIARY orderline_auxiliary;
    STOCK_AUXILIARY stock_auxiliary;
    ITEM_AUXILIARY item_auxiliary;
};
struct PAYMENTQUERY_AUXILIARY
{
    WAREHOUSE_AUXILIARY warehouse_auxiliary;
    DISTRICT_AUXILIARY distrct_auxiliary;
    CUSTOMER_AUXILIARY customer_auxiliary;
    HISTORY_AUXILIARY history_auxiliary;
};
