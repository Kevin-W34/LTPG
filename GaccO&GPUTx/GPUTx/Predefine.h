#include "Random.h"
#include "Database.h"

#define INSERT 1
#define UPDATE 2
#define SELECT 3
#define NEWORDER_TYPE 1
#define PAYMENT_TYPE 2
#define NEWORDERPERCENT (int)0
#define CPU_THREAD_NUM (int)32
#define BATCH_SIZE (int)1048576
#define WAREHOUSE_SIZE (int)64

#define EPOCH (int)5000
#define WARMUP (int)1000

#define CG_SIZE (int)1
#define CG_SIZE_CONST (int)16
#define HASH_SIZE (int)10
#define NORMAL_HASH_SIZE (int)1
#define COMPETITION_HASH_SIZE (int)10
#define CONFIG 64
#define BLOCKSIZE 512
