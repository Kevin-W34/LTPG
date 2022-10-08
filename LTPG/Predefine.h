#include "Random.h"
#include "Database.h"

#define INSERT 1
#define UPDATE 2
#define SELECT 3
#define CPU_THREAD_NUM (int)8

#define NEWORDERPERCENT (int)50
#define BATCH_SIZE (int)65536
#define WAREHOUSE_SIZE (int)8

#define EPOCH (int)60
#define WARMUP (int)10

#define CG_SIZE (int)1
#define CG_SIZE_CONST (int)16
#define HASH_SIZE (int)10
#define NORMAL_HASH_SIZE (int)1
#define COMPETITION_HASH_SIZE (int)10
#define CONFIG 64
#define BLOCKSIZE 32

