#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string>
#include "Database.h"
#include "Predefine.h"
struct NewOrderQuery
{
    int isLocal() const
    {
        for (auto i = 0; i < O_OL_CNT; i++)
        {
            if (INFO[i].OL_SUPPLY_W_ID != W_ID)
            {
                return 0;
            }
        }
        return 1;
    }
    int W_ID;
    int D_ID;
    int C_ID;
    int O_ID;
    int N_O_ID;
    int O_OL_CNT;
    int O_OL_ID;
    struct NewOrderQueryInfo
    {
        int OL_I_ID;
        int OL_SUPPLY_W_ID;
        int OL_QUANTITY;
    };

    NewOrderQueryInfo INFO[15];
};

class MakeNewOrder
{
public:
    int NOID[WAREHOUSE_SIZE];
    int OID[WAREHOUSE_SIZE];
    int OOLID[WAREHOUSE_SIZE];
    void initial()
    {
        for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
        {
            OID[i] = 0;
            OOLID[i] = 0;
            NOID[i] = 0;
        }
        // for (size_t i = 0; i < DISTRICT_SIZE; i++)
        // {
        //     NOID[i] = 0;
        // }
    }

    NewOrderQuery make()
    {
        NewOrderQuery query;
        query.W_ID = (int)random.uniform_dist(0, WAREHOUSE_SIZE - 1); // [0 , WAREHOUSE_SIZE - 1]
        query.D_ID = (int)random.uniform_dist(0, 9);                  // [0 , 9]
        // query.C_ID = (int)random.non_uniform_distribution(1023, 1, 3000) - 1; // [0 , 2999]
        query.C_ID = (int)random.uniform_dist(0, 2999);
        query.O_ID = this->OID[query.W_ID];
        this->OID[query.W_ID] += 1;
        if (this->OID[query.W_ID] >= 30000)
        {
            this->OID[query.W_ID] = 0;
        }

        query.O_OL_CNT = (int)random.uniform_dist(5, 15); // [5 , 15]
        // query.O_OL_CNT = 5;

        query.O_OL_ID = this->OOLID[query.W_ID];
        this->OOLID[query.W_ID] += query.O_OL_CNT;
        if (this->OOLID[query.W_ID] >= 450000)
        {
            this->OOLID[query.W_ID] = 0;
        }

        query.N_O_ID = this->NOID[query.W_ID];
        this->NOID[query.W_ID] += 1;
        if (this->NOID[query.W_ID] >= 30000)
        {
            this->NOID[query.W_ID] = 0;
        }

        for (size_t i = 0; i < query.O_OL_CNT; i++)
        {
            // query.INFO[i].OL_I_ID = (int)random.non_uniform_distribution(8191, 1, 100000) - 1; //[0 , 99999];
            query.INFO[i].OL_I_ID = (int)random.uniform_dist(0, 99999);
            // query.INFO[i].OL_I_ID = (int)random.uniform_dist(0, 9);
            for (size_t k = 0; k < i; k++)
            {
                while (query.INFO[k].OL_I_ID == query.INFO[i].OL_I_ID)
                {
                    // query.INFO[i].OL_I_ID = (int)random.non_uniform_distribution(8191, 1, 100000) - 1;
                    query.INFO[i].OL_I_ID = (int)random.uniform_dist(0, 99999);
                }
            }

            query.INFO[i].OL_SUPPLY_W_ID = query.W_ID; // Home Warehouse
            // int isLocal = random.uniform_dist(1, 100);
            // if (isLocal == 1) // Remote Warehouse
            // {
            //     isLocal = random.uniform_dist(0, 7);
            //     while (isLocal == query.W_ID)
            //     {
            //         isLocal = random.uniform_dist(0, 7);
            //     }
            //     query.INFO[i].OL_SUPPLY_W_ID = isLocal;
            // }
            query.INFO[i].OL_QUANTITY = (int)random.uniform_dist(1, 2);
        }

        return query;
    } /* make new new order */

private:
    Random random;
};

struct Payment
{
    int W_ID;
    int D_ID;
    int C_ID;
    int C_LAST;
    int isName; // 0,id; 1,name
    int C_D_ID;
    int C_W_ID;
    int H_AMOUNT;
    int H_ID;
};

class MakePayment
{
private:
    Random random;

public:
    int HID[WAREHOUSE_SIZE];
    void initial()
    {
        for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
        {
            HID[i] = 0;
        }
    }

    Payment make()
    {
        Payment payment;
        payment.W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        // payment.W_ID = 1;
        payment.D_ID = random.uniform_dist(0, 9);
        int isName = random.uniform_dist(1, 100);
        if (isName < 61)
        {
            payment.isName = 1;
            payment.C_ID = 0;
            payment.C_LAST = random.uniform_dist(0, 15);
        } /* C_LAST */
        else
        {
            payment.isName = 0;
            payment.C_ID = random.non_uniform_distribution(1023, 1, 3000) - 1;
        } /* C_ID */

        // payment.C_ID = random.non_uniform_distribution(1023, 1, 3000) - 1; /* only pay by C_ID */
        int isLocal = random.uniform_dist(1, 100);
        if (isLocal <= 85) // 85
        {
            payment.C_W_ID = payment.W_ID;
        } /* Local */
        else
        {
            payment.C_W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
            while (payment.C_W_ID == payment.W_ID)
            {
                payment.C_W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
            }
        } /* Remote */
        payment.C_D_ID = random.uniform_dist(0, 9);
        payment.H_AMOUNT = random.uniform_dist(1, 5000);
        payment.H_ID = this->HID[payment.W_ID];
        this->HID[payment.W_ID] += 1;
        if (this->HID[payment.W_ID] == 30000)
        {
            this->HID[payment.W_ID] = 0;
        }

        return payment;
    } /* make new payment */
};

struct TestQuery
{
    int info_length;
    struct testQueryInfo
    {
        int column; // 列
        int row;    // 行
        int count;  // 数量
    };
    testQueryInfo info[64];
};

class MakeTestQuery
{
private:
    Random random;

public:
    TestQuery make()
    {
        TestQuery query;
        // query.info_length = random.uniform_dist(55, 64);
        query.info_length = random.uniform_dist(5, 15);
        // query.info_length = 40;
        for (size_t i = 0; i < query.info_length; i++)
        {
            query.info[i].column = random.uniform_dist(0, 9);
            query.info[i].row = random.uniform_dist(0, 6799999);
            query.info[i].count = random.uniform_dist(1, 5);
        }
        return query;
    }
};

struct OrderStatusQuery
{
    int C_ID;
    int C_LAST;
    int isName; // 0,id; 1,name
    int C_D_ID;
    int C_W_ID;
};

class MakeOrderStatus
{
private:
    Random random;

public:
    OrderStatusQuery make()
    {
        OrderStatusQuery orderstatus;
        orderstatus.C_W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        orderstatus.C_D_ID = random.uniform_dist(0, 9);
        orderstatus.isName = random.uniform_dist(1, 100);
        if (orderstatus.isName < 61)
        {
            orderstatus.isName = 1;
            orderstatus.C_ID = 0;
            orderstatus.C_LAST = random.uniform_dist(0, 15);
        }
        else
        {
            orderstatus.isName = 0;
            orderstatus.C_ID = random.non_uniform_distribution(1023, 1, 3000);
        }
        return orderstatus;
    }
};
