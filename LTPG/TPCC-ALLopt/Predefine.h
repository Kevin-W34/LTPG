#pragma once
#include "stdint.h"
#include <mutex>
#include <condition_variable>

const unsigned int SET_DEVICE = (unsigned int)6;
const unsigned int EPOCH_TP = (unsigned int)500;
const unsigned int BATCH_SIZE = (unsigned int)4096;
const unsigned long long WAREHOUSE_SIZE = (unsigned long long)8;
const unsigned int NEWORDER_PERCENT = (unsigned int)100;

// warehouse size
// payment = 100 - NEWORDER_PERCENT
const unsigned int CPU_THREAD_SIZE = (unsigned int)BATCH_SIZE / 2 > 64 ? 64 : BATCH_SIZE / 2;
const unsigned int WARMUP_TP = (unsigned int)(EPOCH_TP * 0.1);
const unsigned int MINI_BATCH_SIZE = (unsigned int)64;
const unsigned int MINI_BATCH_CNT = (unsigned int)(BATCH_SIZE / MINI_BATCH_SIZE);
const unsigned int STREAM_SIZE = (unsigned int)10;

const unsigned int NEWORDER_CNT = (unsigned int)(BATCH_SIZE * NEWORDER_PERCENT / 100);
const unsigned int PAYMENT_CNT = (unsigned int)(BATCH_SIZE * (100 - NEWORDER_PERCENT) / 100);
const unsigned int ORDERSTATUS_CNT = (unsigned int)(BATCH_SIZE / 128);
const unsigned int DELIVERY_CNT = (unsigned int)(BATCH_SIZE / 128);
const unsigned int STOCK_CNT = (unsigned int)(BATCH_SIZE / 128);
const unsigned int OP_CNT = (unsigned int)(50 * NEWORDER_CNT + 7 * PAYMENT_CNT);
const unsigned int MINI_NEWORDER_CNT = (unsigned int)(MINI_BATCH_SIZE * NEWORDER_PERCENT / 100);
const unsigned int MINI_PAYMENT_CNT = (unsigned int)(MINI_BATCH_SIZE * (100 - NEWORDER_PERCENT) / 100);
const unsigned int MINI_OP_CNT = (unsigned int)(50 * MINI_NEWORDER_CNT + 7 * MINI_PAYMENT_CNT);
const unsigned int PRE_GEN_EPOCH = (unsigned int)(EPOCH_TP * 0.1) > 100 ? 100 : (EPOCH_TP * 0.1);
// const unsigned int PRE_GEN_EPOCH = (unsigned int)(EPOCH_TP * 0.1);

const unsigned int WARP_SIZE = (unsigned int)32;
const unsigned int GRID_SIZE = (unsigned int)512;
const unsigned int BLOCK_SIZE = (unsigned int)512;
const unsigned int EXECUTE_WARP = (unsigned int)147;
const unsigned int CHECK_WARP = (unsigned int)93;
const unsigned int WRITEBACK_WARP = (unsigned int)67;
const unsigned int EXECUTE_BLOCK_SIZE = (unsigned int)512;
const unsigned int EXECUTE_GRID_SIZE = (unsigned int)EXECUTE_WARP * BATCH_SIZE / EXECUTE_BLOCK_SIZE + 1;
const unsigned int CHECK_BLOCK_SIZE = (unsigned int)512;
const unsigned int CHECK_GRID_SIZE = (unsigned int)CHECK_WARP * BATCH_SIZE / CHECK_BLOCK_SIZE + 1;
const unsigned int WRITEBACK_BLOCK_SIZE = (unsigned int)512;
const unsigned int WRITEBACK_GRID_SIZE = (unsigned int)WRITEBACK_WARP * BATCH_SIZE / WRITEBACK_BLOCK_SIZE + 1;

const unsigned long long DISTRICT_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 10);
const unsigned long long CUSTOMER_SIZE = (unsigned long long)(DISTRICT_SIZE * 3000);
const unsigned long long HISTORY_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 30000);
const unsigned long long NEWORDER_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 30000);
const unsigned long long ORDER_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 30000);
const unsigned long long ORDERLINE_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 450000);
const unsigned long long STOCK_SIZE = (unsigned long long)(WAREHOUSE_SIZE * 100000);
const unsigned long long ITEM_SIZE = (unsigned long long)100000;

const unsigned int WAREHOUSE_COLUMN = (unsigned int)9;
const unsigned int DISTRICT_COLUMN = (unsigned int)11;
const unsigned int CUSTOMER_COLUMN = (unsigned int)21;
const unsigned int HISTORY_COLUMN = (unsigned int)5;
const unsigned int NEWORDER_COLUMN = (unsigned int)3;
const unsigned int ORDER_COLUMN = (unsigned int)8;
const unsigned int ORDERLINE_COLUMN = (unsigned int)10;
const unsigned int STOCK_COLUMN = (unsigned int)17;
const unsigned int ITEM_COLUMN = (unsigned int)5;

const unsigned int TABLE_SIZE_1D = (unsigned int)(WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + ITEM_SIZE);

const unsigned int WAREHOUSE_ID = (unsigned int)0;
const unsigned int DISTRICT_ID = (unsigned int)1;
const unsigned int CUSTOMER_ID = (unsigned int)2;
const unsigned int HISTORY_ID = (unsigned int)3;
const unsigned int NEWORDER_ID = (unsigned int)4;
const unsigned int ORDER_ID = (unsigned int)5;
const unsigned int ORDERLINE_ID = (unsigned int)6;
const unsigned int STOCK_ID = (unsigned int)7;
const unsigned int ITEM_ID = (unsigned int)8;

// 252
const unsigned int NEWORDER_COLUMN_0 = (unsigned int)0b100;             // 1
const unsigned int NEWORDER_COLUMN_1 = (unsigned int)0b101000;          // 2
const unsigned int NEWORDER_COLUMN_2 = (unsigned int)0b100100000100000; // 3
const unsigned int NEWORDER_COLUMN_3 = (unsigned int)0b1110;            // 3
const unsigned int NEWORDER_COLUMN_4 = (unsigned int)0b1110;            // 3
const unsigned int NEWORDER_COLUMN_5 = (unsigned int)0b111000;          // 3
const unsigned int NEWORDER_COLUMN_6 = (unsigned int)0b111000;          // 3
const unsigned int NEWORDER_COLUMN_7 = (unsigned int)0b11111111110;     // 10

// 34
const unsigned int PAYMENT_COLUMN_0 = (unsigned int)0b1111110000;             // 6
const unsigned int PAYMENT_COLUMN_1 = (unsigned int)0b1000;                   // 1
const unsigned int PAYMENT_COLUMN_2 = (unsigned int)0b111111000000;           // 6
const unsigned int PAYMENT_COLUMN_3 = (unsigned int)0b10000;                  // 1
const unsigned int PAYMENT_COLUMN_4 = (unsigned int)0b1111111111000001110000; // 13
const unsigned int PAYMENT_COLUMN_5 = (unsigned int)0b10001000000;            // 2
const unsigned int PAYMENT_COLUMN_6 = (unsigned int)0b111110;                 // 5

const unsigned int READ_TYPE = (unsigned int)0b0;
const unsigned int WRITE_TYPE = (unsigned int)0b1;

const unsigned int NEWORDER_TYPE_CNT = (unsigned int)8;
const unsigned int PAYMENT_TYPE_CNT = (unsigned int)7;
const unsigned int ORDERSTATUS_TYPE_CNT = (unsigned int)4;
const unsigned int NEWORDER_WB_TYPE_CNT = (unsigned int)4;
const unsigned int PAYMENT_WB_TYPE_CNT = (unsigned int)4;

const unsigned int OP_TYPE_CNT = (unsigned int)(NEWORDER_TYPE_CNT * NEWORDER_CNT + PAYMENT_TYPE_CNT * PAYMENT_CNT);

const unsigned int NEWORDER_OP_CNT = (unsigned int)50;
const unsigned int PAYMENT_OP_CNT = (unsigned int)7;
const unsigned int ORDERSTATUS_OP_CNT = (unsigned int)4;
const unsigned int DELIVERY_OP_CNT = (unsigned int)7;
const unsigned int STOCK_OP_CNT = (unsigned int)20;

const unsigned int NEWORDER_WARP_0 = (unsigned int)(NEWORDER_CNT / 32 + 1);
const unsigned int PAYMENT_WARP_0 = (unsigned int)(PAYMENT_CNT / 32 + 1);

std::mutex mtx_start;
std::mutex mtx_end;
std::unique_lock<std::mutex> lock_start(mtx_start);
std::unique_lock<std::mutex> lock_end(mtx_end);
std::condition_variable cv_start_execute;
std::condition_variable cv_end_execute;
bool bool_make = false;
bool bool_execute = true;
