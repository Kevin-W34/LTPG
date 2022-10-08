#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "Query.h"
#include "Predefine.h"


#define DISTRICT_SIZE (int)(WAREHOUSE_SIZE * 10)
#define CUSTOMER_SIZE (int)(DISTRICT_SIZE * 3000)
#define HISTORY_SIZE (int)(WAREHOUSE_SIZE * 30000)
#define STOCK_SIZE (int)(WAREHOUSE_SIZE * 100000)
#define NEWORDER_SIZE (int)(WAREHOUSE_SIZE * 30000)
#define ITEM_SIZE (int)100000
#define ORDERLINE_SIZE (int)(WAREHOUSE_SIZE * 450000)
#define ORDER_SIZE (int)(WAREHOUSE_SIZE * 30000)
#define TABLE_SIZE_2D (int)(1 + 10 + 30000 + 30000 + 100000 + 30000 + 450000 + 30000)
#define DATA_SIZE (int)21

#define TABLE_SIZE_1D (int)(WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + ITEM_SIZE)

struct WAREHOUSE /* 100000 Items per Warehouse and all warehouses have same items */
{
    int W_ID;       // 0
    int W_TAX;      // 1
    int W_YTD;      // 2
    int W_NAME;     // 3
    int W_STREET_1; // 4
    int W_STREET_2; // 5
    int W_CITY;     // 6
    int W_STATE;    // 7
    int W_ZIP;      // 8
};

struct DISTRICT /* 10 District per Warehouse */
{
    int D_ID;        // 0
    int D_W_ID;      // 1
    int D_TAX;       // 2
    int D_YTD;       // 3
    int D_NEXT_O_ID; // 4
    int D_NAME;      // 5
    int D_STREET_1;  // 6
    int D_STREET_2;  // 7
    int D_CITY;      // 8
    int D_STATE;     // 9
    int D_ZIP;       // 10
};

struct CUSTOMER /* 3000 Costomers per District */
{
    int C_ID;           // 0
    int C_W_ID;         // 1
    int C_D_ID;         // 2
    int C_CREDIT_LIM;   // 3
    int C_DISCOUNT;     // 4
    int C_BALANCE;      // 5
    int C_YTD_PAYMENT;  // 6
    int C_PAYMENT_CNT;  // 7
    int C_DELIVERY_CNT; // 8
    int C_DATA;         // 9
    int C_CREDIT;       // 10
    int C_FIRST;        // 11
    int C_MIDDLE;       // 12
    int C_LAST;         // 13
    int C_STREET_1;     // 14
    int C_STREET_2;     // 15
    int C_CITY;         // 16
    int C_STATE;        // 17
    int C_ZIP;          // 18
    int C_PHONE;        // 19
    int C_SINCE;        // 20
};

struct HISTORY
{
    int H_C_ID;   // 0
    int H_C_D_ID; // 1
    int H_C_W_ID; // 2
    int H_D_ID;   // 3
    int H_W_ID;   // 4
};

struct NEWORDER
{
    int NO_O_ID; // 0  /* Foreign Key O_ID */
    int NO_D_ID; // 1  /* Foreign Key O_D_ID */
    int NO_W_ID; // 2  /* Foreign Key O_W_ID */
};

struct ORDER /*  */
{
    int O_ID;         // 0
    int O_D_ID;       // 1  /* Froeign Key D_ID */
    int O_W_ID;       // 2  /* Foreign Key W_ID */
    int O_C_ID;       // 3  /* Foreign Key C_ID */
    int O_ENTRY_D;    // 4
    int O_CARRIER_ID; // 5
    int O_OL_CNT;     // 6
    int O_ALL_LOCAL;  // 7
};

struct ORDERLINE
{
    int OL_O_ID;        // 0  /* Foreign Key O_ID */
    int OL_D_ID;        // 1  /* Foreign Key D_ID */
    int OL_W_ID;        // 2  /* Foreign Key W_ID */
    int OL_NUMBER;      // 3
    int OL_I_ID;        // 4  /* Froeign Key S_I_ID; */
    int OL_SUPPLY_W_ID; // 5  /* Foreign Key S_W_ID; */
    int OL_DELIVERY_D;  // 6
    int OL_QUANLITY;    // 7
    int OL_AMOUNT;      // 8
    int OL_DIST_INF;    // 9
};

struct ITEM
{
    int I_ID;    // 0
    int I_IM_ID; // 1
    int I_PRICE; // 2
    int I_NAME;  // 3
    int I_DATA;  // 4
};

struct STOCK
{
    int S_I_ID;       // 0  /* Foreign Key I_ID */
    int S_W_ID;       // 1  /* Foreign Key W_ID */
    int S_QUANTITY;   // 2
    int S_YTD;        // 3
    int S_ORDER_CNT;  // 4
    int S_REMOVE_CNT; // 5
    int S_DIST_01;    // 6
    int S_DIST_02;    // 7
    int S_DIST_03;    // 8
    int S_DIST_04;    // 9
    int S_DIST_05;    // 10
    int S_DIST_06;    // 11
    int S_DIST_07;    // 12
    int S_DIST_08;    // 13
    int S_DIST_09;    // 14
    int S_DIST_10;    // 15
    int S_DATA;       // 16
};

struct GLOBAL
{
    int data[21] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct GLOBAL_for_2D
{
    // int data[21] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int data0 = 0;
    int data1 = 0;
    int data2 = 0;
    int data3 = 0;
    int data4 = 0;
    int data5 = 0;
    int data6 = 0;
    int data7 = 0;
    int data8 = 0;
    int data9 = 0;
    int data10 = 0;
    int data11 = 0;
    int data12 = 0;
    int data13 = 0;
    int data14 = 0;
    int data15 = 0;
    int data16 = 0;
    int data17 = 0;
    int data18 = 0;
    int data19 = 0;
    int data20 = 0;
};

class Database
{
private:
    std::vector<WAREHOUSE> warehouse_tbl; /* warehouse table */
    std::vector<DISTRICT> district_tbl;   /* district table */
    std::vector<CUSTOMER> customer_tbl;   /* customer table */
    std::vector<HISTORY> history_tbl;     /* history table */
    std::vector<NEWORDER> neworder_tbl;   /* neworder table */
    std::vector<ORDER> order_tbl;         /* order table */
    std::vector<ORDERLINE> orderline_tbl; /* orderline table */
    std::vector<ITEM> item_tbl;           /* item table */
    std::vector<STOCK> stock_tbl;         /* stock table */
    Random random;

public:
    int neworder_ID = 0;
    int orderline_ID = 0;
    int order_ID = 0;
    int history_ID = 0;
    // int TABLE_SIZE_1D = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + ITEM_SIZE;
    // int TABLE_SIZE_2D = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE;
    // int TABLE_SIZE_2D = 1 + 10 + 30000 + 30000 + 100000 + 30000 + 450000 + 30000;
    void initial_data()
    {
        warehouse_tbl = std::vector<WAREHOUSE>(WAREHOUSE_SIZE);
        district_tbl = std::vector<DISTRICT>(DISTRICT_SIZE);
        customer_tbl = std::vector<CUSTOMER>(CUSTOMER_SIZE);
        history_tbl = std::vector<HISTORY>(HISTORY_SIZE);
        neworder_tbl = std::vector<NEWORDER>(NEWORDER_SIZE);
        order_tbl = std::vector<ORDER>(ORDER_SIZE);
        orderline_tbl = std::vector<ORDERLINE>(ORDERLINE_SIZE);
        item_tbl = std::vector<ITEM>(ITEM_SIZE);
        stock_tbl = std::vector<STOCK>(STOCK_SIZE);

        std::cout << "WAREHOUSE_SIZE = " << WAREHOUSE_SIZE << std::endl;
        std::cout << "DISTRICT_SIZE = " << DISTRICT_SIZE << std::endl;
        std::cout << "CUSTOMER_SIZE = " << CUSTOMER_SIZE << std::endl;
        std::cout << "HISTORY_SIZE = " << HISTORY_SIZE << std::endl;
        std::cout << "NEWORDER_SIZE = " << NEWORDER_SIZE << std::endl;
        std::cout << "ORDER_SIZE = " << ORDER_SIZE << std::endl;
        std::cout << "ORDERLINE_SIZE = " << ORDERLINE_SIZE << std::endl;
        std::cout << "STOCK_SIZE = " << STOCK_SIZE << std::endl;
        std::cout << "ITEM_SIZE = " << ITEM_SIZE << std::endl;
        std::cout << std::endl;
        std::cout << "TABLE_SIZE = " << TABLE_SIZE_1D << std::endl;
        std::cout << "======================================\n";
        std::cout << std::endl;

        for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
        {
            warehouse_tbl[i].W_ID = i;
            warehouse_tbl[i].W_TAX = 1;
            warehouse_tbl[i].W_YTD = 1;
            warehouse_tbl[i].W_NAME = 0;
            warehouse_tbl[i].W_STREET_1 = 0;
            warehouse_tbl[i].W_STREET_2 = 0;
            warehouse_tbl[i].W_CITY = 10;
            warehouse_tbl[i].W_STATE = 0;
            warehouse_tbl[i].W_ZIP = 0;
        }
        for (size_t j = 0; j < DISTRICT_SIZE; j++)
        {
            district_tbl[j].D_ID = j % 10;
            district_tbl[j].D_W_ID = j / 10;
            district_tbl[j].D_TAX = 1;
            district_tbl[j].D_YTD = 1;
            district_tbl[j].D_NEXT_O_ID = 0;
            district_tbl[j].D_NAME = 0;
            district_tbl[j].D_STREET_1 = 0;
            district_tbl[j].D_STREET_2 = 0;
            district_tbl[j].D_CITY = 0;
            district_tbl[j].D_STATE = 0;
            district_tbl[j].D_ZIP = 0;
        }
        for (size_t k = 0; k < CUSTOMER_SIZE; k++)
        {
            customer_tbl[k].C_ID = k % 3000;
            customer_tbl[k].C_W_ID = k / 30000;
            customer_tbl[k].C_D_ID = (k / 3000) % 10;
            customer_tbl[k].C_CREDIT_LIM = 100000;
            customer_tbl[k].C_DISCOUNT = 100000;
            customer_tbl[k].C_BALANCE = 100000;
            customer_tbl[k].C_YTD_PAYMENT = 100000;
            customer_tbl[k].C_PAYMENT_CNT = 100000;
            customer_tbl[k].C_DELIVERY_CNT = 100000;
            customer_tbl[k].C_DATA = 100000;
            customer_tbl[k].C_CREDIT = random.uniform_dist(0, 1); // 0 bad; 1 good
            customer_tbl[k].C_FIRST = 0;
            customer_tbl[k].C_MIDDLE = 0;
            customer_tbl[k].C_LAST = random.uniform_dist(0, 15);
            customer_tbl[k].C_STREET_1 = 0;
            customer_tbl[k].C_STREET_2 = 0;
            customer_tbl[k].C_CITY = 0;
            customer_tbl[k].C_STATE = 0;
            customer_tbl[k].C_ZIP = 0;
            customer_tbl[k].C_PHONE = 0;
            customer_tbl[k].C_SINCE = 0;
        }

        for (size_t j = 0; j < STOCK_SIZE; j++)
        {
            stock_tbl[j].S_I_ID = j % 100000;
            stock_tbl[j].S_W_ID = j / 100000;
            stock_tbl[j].S_QUANTITY = 100000;
            stock_tbl[j].S_YTD = 100000;
            stock_tbl[j].S_ORDER_CNT = 0;
            stock_tbl[j].S_REMOVE_CNT = 0;
            stock_tbl[j].S_DIST_01 = 1;
            stock_tbl[j].S_DIST_02 = 2;
            stock_tbl[j].S_DIST_03 = 3;
            stock_tbl[j].S_DIST_04 = 4;
            stock_tbl[j].S_DIST_05 = 5;
            stock_tbl[j].S_DIST_06 = 6;
            stock_tbl[j].S_DIST_07 = 7;
            stock_tbl[j].S_DIST_08 = 8;
            stock_tbl[j].S_DIST_09 = 9;
            stock_tbl[j].S_DIST_10 = 10;
            stock_tbl[j].S_DATA = 0;
        }

        for (size_t i = 0; i < ITEM_SIZE; i++)
        {
            item_tbl[i].I_ID = i;
            item_tbl[i].I_IM_ID = 1000;
            item_tbl[i].I_PRICE = 1;
            item_tbl[i].I_DATA = 0;
            item_tbl[i].I_NAME = 0;
        }
    }

    GLOBAL *getSnapShot_2D()
    {
        // int size = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + ITEM_SIZE;
        std::cout << "start snapshot\n";

        GLOBAL *current_2D;
        current_2D = new GLOBAL[TABLE_SIZE_1D];

        int loc = 0;
        for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
        {
            current_2D[i].data[0] = warehouse_tbl[i].W_ID;
            current_2D[i].data[1] = warehouse_tbl[i].W_TAX;
            current_2D[i].data[2] = warehouse_tbl[i].W_YTD;
            current_2D[i].data[3] = warehouse_tbl[i].W_NAME;
            current_2D[i].data[4] = warehouse_tbl[i].W_STREET_1;
            current_2D[i].data[5] = warehouse_tbl[i].W_STREET_2;
            current_2D[i].data[6] = warehouse_tbl[i].W_CITY;
            current_2D[i].data[7] = warehouse_tbl[i].W_STATE;
            current_2D[i].data[8] = warehouse_tbl[i].W_ZIP;
        }
        loc += WAREHOUSE_SIZE;
        for (size_t j = 0; j < DISTRICT_SIZE; j++)
        {
            // current_2D[j + loc] = new int[DATA_SIZE];
            current_2D[j + loc].data[0] = district_tbl[j].D_ID;
            current_2D[j + loc].data[1] = district_tbl[j].D_W_ID;
            current_2D[j + loc].data[2] = district_tbl[j].D_TAX;
            current_2D[j + loc].data[3] = district_tbl[j].D_YTD;
            current_2D[j + loc].data[4] = district_tbl[j].D_NEXT_O_ID;
            current_2D[j + loc].data[5] = district_tbl[j].D_NAME;
            current_2D[j + loc].data[6] = district_tbl[j].D_STREET_1;
            current_2D[j + loc].data[7] = district_tbl[j].D_STREET_2;
            current_2D[j + loc].data[8] = district_tbl[j].D_CITY;
            current_2D[j + loc].data[9] = district_tbl[j].D_STATE;
            current_2D[j + loc].data[10] = district_tbl[j].D_ZIP;
        }
        loc += DISTRICT_SIZE; // district
        for (size_t j = 0; j < CUSTOMER_SIZE; j++)
        {
            current_2D[j + loc].data[0] = customer_tbl[j].C_ID;
            current_2D[j + loc].data[1] = customer_tbl[j].C_W_ID;
            current_2D[j + loc].data[2] = customer_tbl[j].C_D_ID;
            current_2D[j + loc].data[3] = customer_tbl[j].C_CREDIT_LIM;
            current_2D[j + loc].data[4] = customer_tbl[j].C_DISCOUNT;
            current_2D[j + loc].data[5] = customer_tbl[j].C_BALANCE;
            current_2D[j + loc].data[6] = customer_tbl[j].C_YTD_PAYMENT;
            current_2D[j + loc].data[7] = customer_tbl[j].C_PAYMENT_CNT;
            current_2D[j + loc].data[8] = customer_tbl[j].C_DELIVERY_CNT;
            current_2D[j + loc].data[9] = customer_tbl[j].C_DATA;
            current_2D[j + loc].data[10] = customer_tbl[j].C_CREDIT;
            current_2D[j + loc].data[11] = customer_tbl[j].C_FIRST;
            current_2D[j + loc].data[12] = customer_tbl[j].C_MIDDLE;
            current_2D[j + loc].data[13] = customer_tbl[j].C_LAST;
            current_2D[j + loc].data[14] = customer_tbl[j].C_STREET_1;
            current_2D[j + loc].data[15] = customer_tbl[j].C_STREET_2;
            current_2D[j + loc].data[16] = customer_tbl[j].C_CITY;
            current_2D[j + loc].data[17] = customer_tbl[j].C_STATE;
            current_2D[j + loc].data[18] = customer_tbl[j].C_ZIP;
            current_2D[j + loc].data[19] = customer_tbl[j].C_PHONE;
            current_2D[j + loc].data[20] = customer_tbl[j].C_SINCE;
        }
        loc += CUSTOMER_SIZE;                   // customer
        loc += HISTORY_SIZE;                    // history
        loc += NEWORDER_SIZE;                   // neworder
        loc += ORDER_SIZE;                      // order
        loc += ORDERLINE_SIZE;                  // orderline
        for (size_t j = 0; j < STOCK_SIZE; j++) // stock
        {
            current_2D[j + loc].data[0] = stock_tbl[j].S_I_ID;
            current_2D[j + loc].data[1] = stock_tbl[j].S_W_ID;
            current_2D[j + loc].data[2] = stock_tbl[j].S_QUANTITY;
            current_2D[j + loc].data[3] = stock_tbl[j].S_YTD;
            current_2D[j + loc].data[4] = stock_tbl[j].S_ORDER_CNT;
            current_2D[j + loc].data[5] = stock_tbl[j].S_DIST_01;
            current_2D[j + loc].data[6] = stock_tbl[j].S_DIST_02;
            current_2D[j + loc].data[7] = stock_tbl[j].S_DIST_03;
            current_2D[j + loc].data[8] = stock_tbl[j].S_DIST_04;
            current_2D[j + loc].data[9] = stock_tbl[j].S_DIST_05;
            current_2D[j + loc].data[10] = stock_tbl[j].S_DIST_06;
            current_2D[j + loc].data[11] = stock_tbl[j].S_DIST_07;
            current_2D[j + loc].data[12] = stock_tbl[j].S_DIST_08;
            current_2D[j + loc].data[13] = stock_tbl[j].S_DIST_09;
            current_2D[j + loc].data[14] = stock_tbl[j].S_DIST_10;
            current_2D[j + loc].data[15] = stock_tbl[j].S_DATA;
        }
        loc += STOCK_SIZE;
        for (size_t j = 0; j < ITEM_SIZE; j++) // item
        {
            current_2D[j + loc].data[0] = item_tbl[j].I_ID;
            current_2D[j + loc].data[1] = item_tbl[j].I_IM_ID;
            current_2D[j + loc].data[2] = item_tbl[j].I_PRICE;
            current_2D[j + loc].data[3] = item_tbl[j].I_NAME;
            current_2D[j + loc].data[4] = item_tbl[j].I_DATA;
        }
        std::cout << "finish snapshot\n";
        return current_2D;
    }

    GLOBAL *getSnapShot_1D()
    {
        GLOBAL *current_1D;
        current_1D = new GLOBAL[TABLE_SIZE_1D];
        int flag = 0;
        int loc = 0;
        for (size_t i = flag; i < flag + WAREHOUSE_SIZE; i++)
        {
            current_1D[i].data[0] = warehouse_tbl[loc].W_ID;
            current_1D[i].data[1] = warehouse_tbl[loc].W_TAX;
            current_1D[i].data[2] = warehouse_tbl[loc].W_YTD;
            loc += 1;
        }
        flag += WAREHOUSE_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + DISTRICT_SIZE; i++)
        {
            current_1D[i].data[0] = district_tbl[loc].D_ID;
            current_1D[i].data[1] = district_tbl[loc].D_W_ID;
            current_1D[i].data[2] = district_tbl[loc].D_TAX;
            current_1D[i].data[3] = district_tbl[loc].D_YTD;
            current_1D[i].data[4] = district_tbl[loc].D_NEXT_O_ID;
            loc += 1;
        }
        flag += DISTRICT_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + CUSTOMER_SIZE; i++)
        {
            current_1D[i].data[0] = customer_tbl[loc].C_ID;
            current_1D[i].data[1] = customer_tbl[loc].C_W_ID;
            current_1D[i].data[2] = customer_tbl[loc].C_D_ID;
            current_1D[i].data[3] = customer_tbl[loc].C_CREDIT_LIM;
            current_1D[i].data[4] = customer_tbl[loc].C_DISCOUNT;
            current_1D[i].data[5] = customer_tbl[loc].C_BALANCE;
            current_1D[i].data[6] = customer_tbl[loc].C_YTD_PAYMENT;
            current_1D[i].data[7] = customer_tbl[loc].C_PAYMENT_CNT;
            current_1D[i].data[8] = customer_tbl[loc].C_DELIVERY_CNT;
            current_1D[i].data[9] = customer_tbl[loc].C_DATA;
            current_1D[i].data[10] = customer_tbl[loc].C_CREDIT;
            current_1D[i].data[11] = customer_tbl[loc].C_LAST;
            loc += 1;
        }
        flag += CUSTOMER_SIZE;
        flag += HISTORY_SIZE;
        flag += NEWORDER_SIZE;
        flag += ORDER_SIZE;
        flag += ORDERLINE_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + ITEM_SIZE; i++)
        {
            current_1D[i].data[0] = item_tbl[loc].I_ID;
            current_1D[i].data[1] = item_tbl[loc].I_IM_ID;
            current_1D[i].data[2] = item_tbl[loc].I_PRICE;
            loc += 1;
        }
        flag += ITEM_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + STOCK_SIZE; i++)
        {
            current_1D[i].data[0] = stock_tbl[loc].S_I_ID;
            current_1D[i].data[1] = stock_tbl[loc].S_W_ID;
            current_1D[i].data[2] = stock_tbl[loc].S_QUANTITY;
            current_1D[i].data[3] = stock_tbl[loc].S_YTD;
            current_1D[i].data[4] = stock_tbl[loc].S_ORDER_CNT;
            current_1D[i].data[5] = stock_tbl[loc].S_REMOVE_CNT;
            loc += 1;
        }
        return current_1D;
    }

    void refreshSnapShot_1D(GLOBAL *newIndex)
    {
        /* analyse and writeback */
        int flag = 0;
        int loc = 0;
        for (size_t i = flag; i < flag + WAREHOUSE_SIZE; i++)
        {
            warehouse_tbl[loc].W_ID = newIndex[i].data[0];
            warehouse_tbl[loc].W_TAX = newIndex[i].data[1];
            warehouse_tbl[loc].W_YTD = newIndex[i].data[2];
            loc += 1;
        }
        flag += WAREHOUSE_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + DISTRICT_SIZE; i++)
        {
            district_tbl[loc].D_ID = newIndex[i].data[0];
            district_tbl[loc].D_W_ID = newIndex[i].data[1];
            district_tbl[loc].D_TAX = newIndex[i].data[2];
            district_tbl[loc].D_YTD = newIndex[i].data[3];
            district_tbl[loc].D_NEXT_O_ID = newIndex[i].data[4];
            loc += 1;
        }
        flag += DISTRICT_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + CUSTOMER_SIZE; i++)
        {
            customer_tbl[loc].C_ID = newIndex[i].data[0];
            customer_tbl[loc].C_W_ID = newIndex[i].data[1];
            customer_tbl[loc].C_D_ID = newIndex[i].data[2];
            customer_tbl[loc].C_CREDIT_LIM = newIndex[i].data[3];
            customer_tbl[loc].C_DISCOUNT = newIndex[i].data[4];
            customer_tbl[loc].C_BALANCE = newIndex[i].data[5];
            customer_tbl[loc].C_YTD_PAYMENT = newIndex[i].data[6];
            customer_tbl[loc].C_PAYMENT_CNT = newIndex[i].data[7];
            customer_tbl[loc].C_DELIVERY_CNT = newIndex[i].data[8];
            customer_tbl[loc].C_DATA = newIndex[i].data[9];
            customer_tbl[loc].C_CREDIT = newIndex[i].data[10];
            customer_tbl[loc].C_LAST = newIndex[i].data[11];
            loc += 1;
        }
        flag += CUSTOMER_SIZE;
        flag += HISTORY_SIZE;
        flag += NEWORDER_SIZE;
        flag += ORDER_SIZE;
        flag += ORDERLINE_SIZE;
        loc = 0;

        for (size_t i = flag; i < flag + ITEM_SIZE; i++)
        {
            item_tbl[loc].I_ID = newIndex[i].data[0];
            item_tbl[loc].I_IM_ID = newIndex[i].data[1];
            item_tbl[loc].I_PRICE = newIndex[i].data[2];
            loc += 1;
        }
        flag += ITEM_SIZE;
        loc = 0;
        for (size_t i = flag; i < flag + STOCK_SIZE; i++)
        {
            stock_tbl[loc].S_I_ID = newIndex[i].data[0];
            stock_tbl[loc].S_W_ID = newIndex[i].data[1];
            stock_tbl[loc].S_QUANTITY = newIndex[i].data[2];
            stock_tbl[loc].S_YTD = newIndex[i].data[3];
            stock_tbl[loc].S_ORDER_CNT = newIndex[i].data[4];
            stock_tbl[loc].S_REMOVE_CNT = newIndex[i].data[5];
            loc += 1;
        }
    }
    void refreshSnapShot_2D(GLOBAL **newIndex)
    {
    }
};
