#pragma once
#include "stdint.h"
const unsigned int ZIPFIAN = (unsigned int)1;
const unsigned int UNIFORM = (unsigned int)2;

const unsigned int SET_DEVICE =(unsigned int)1;
const unsigned int EPOCH_TP =(unsigned int)5000;
const unsigned int BATCH_SIZE =(unsigned int)10000;
const unsigned long long YCSB_SIZE =(unsigned long long)1000000;
const unsigned int YCSB_OP_SIZE =(unsigned int)10;
const unsigned int YCSB_READ_SIZE =(unsigned int)9;
const unsigned int DATA_DISTRIBUTION =(unsigned int)UNIFORM;
const unsigned int YCSB_SCAN_SIZE =(unsigned int)10;

const unsigned int YCSB_CPU_THREAD_SIZE = (unsigned int)100;
const unsigned int WARMUP_TP = (unsigned int)(EPOCH_TP * 0.1);
const unsigned int MINI_BATCH_SIZE = (unsigned int)64;
const unsigned int MINI_BATCH_CNT = (unsigned int)(BATCH_SIZE / MINI_BATCH_SIZE);
const unsigned int NEWORDER_PERCENT = (unsigned int)50;
const unsigned int STREAM_SIZE = (unsigned int)10;

const unsigned int YCSB_COLUMN = (unsigned int)10;
const unsigned int YCSB_WRITE_SIZE = (unsigned int)YCSB_OP_SIZE - YCSB_READ_SIZE;
const unsigned int YCSB_A_SIZE = BATCH_SIZE;
const unsigned int PRE_GEN_EPOCH = (unsigned int)(EPOCH_TP * 0.1) > 100 ? 100 : (EPOCH_TP * 0.1);
// const unsigned int PRE_GEN_EPOCH = (unsigned int)(EPOCH_TP * 0.1);

const unsigned int WARP_SIZE = (unsigned int)32;
const unsigned int GRID_SIZE = (unsigned int)512;
const unsigned int BLOCK_SIZE = (unsigned int)512;
const unsigned int EXECUTE_BLOCK_SIZE = (unsigned int)512;
const unsigned int EXECUTE_GRID_SIZE = (unsigned int)(YCSB_OP_SIZE * 2) * BATCH_SIZE / EXECUTE_BLOCK_SIZE + 1;
const unsigned int CHECK_BLOCK_SIZE = (unsigned int)512;
const unsigned int CHECK_GRID_SIZE = (unsigned int)(YCSB_READ_SIZE + 2 * YCSB_WRITE_SIZE) * BATCH_SIZE / CHECK_BLOCK_SIZE + 1;
const unsigned int WRITEBACK_BLOCK_SIZE = (unsigned int)512;
const unsigned int WRITEBACK_GRID_SIZE = (unsigned int)(1 + YCSB_WRITE_SIZE) * BATCH_SIZE / WRITEBACK_BLOCK_SIZE + 1;

const unsigned int READ_TYPE = (unsigned int)0b0;
const unsigned int WRITE_TYPE = (unsigned int)0b1;
