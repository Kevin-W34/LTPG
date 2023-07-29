#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "Predefine.h"
#include "Datastructure.h"
#include "Genericfunction.h"
#include "Random.h"
class Database
{
private:
    WAREHOUSE *warehouse_tbl; // warehouse table
    DISTRICT *district_tbl;   // district table
    CUSTOMER *customer_tbl;   // customer table
    HISTORY *history_tbl;     // history table
    NEWORDER *neworder_tbl;   // neworder table
    ORDER *order_tbl;         // order table
    ORDERLINE *orderline_tbl; // orderline table
    STOCK *stock_tbl;         // stock table
    ITEM *item_tbl;           // item table

    unsigned int *customer_name_index; // 客户名字索引

    Random random;

public:
    SNAPSHOT *snapshot;
    SNAPSHOT *snapshot_d;
    INDEX *index;
    INDEX *index_d;
    LOG *log;
    float time;
    Database()
    {

        cudaMallocManaged((void **)&snapshot, sizeof(SNAPSHOT));
        cudaMallocManaged((void **)&index, sizeof(INDEX));
        cudaMalloc((void **)&snapshot_d, sizeof(SNAPSHOT));
        cudaMalloc((void **)&index_d, sizeof(INDEX));
        // cudaMalloc((void **)&log, sizeof(LOG));
        warehouse_tbl = new WAREHOUSE[WAREHOUSE_SIZE];
        district_tbl = new DISTRICT[DISTRICT_SIZE];
        customer_tbl = new CUSTOMER[CUSTOMER_SIZE];
        customer_name_index = new unsigned int[CUSTOMER_SIZE];
        history_tbl = new HISTORY[HISTORY_SIZE];
        neworder_tbl = new NEWORDER[NEWORDER_SIZE];
        order_tbl = new ORDER[ORDER_SIZE];
        orderline_tbl = new ORDERLINE[ORDERLINE_SIZE];
        stock_tbl = new STOCK[STOCK_SIZE];
        item_tbl = new ITEM[ITEM_SIZE];

        clear_LOG();
    }

    ~Database()
    {
        cudaFree(snapshot);
        cudaFree(index);
        cudaFree(snapshot_d);
        cudaFree(index_d);
        // cudaFree(log);

        free(warehouse_tbl);
        free(district_tbl);
        free(customer_tbl);
        free(history_tbl);
        free(neworder_tbl);
        free(order_tbl);
        free(orderline_tbl);
        free(stock_tbl);
        free(item_tbl);
    }

    void clear_LOG()
    {
        // cudaMemset(log->LOG_WAREHOUSE_R, 0xffffffff, sizeof(unsigned int) * WAREHOUSE_SIZE * 2);
        // cudaMemset(log->LOG_WAREHOUSE_W, 0xffffffff, sizeof(unsigned int) * WAREHOUSE_SIZE * 2);
        // cudaMemset(log->LOG_DISTRICT_R, 0xffffffff, sizeof(unsigned int) * DISTRICT_SIZE * 2);
        // cudaMemset(log->LOG_DISTRICT_W, 0xffffffff, sizeof(unsigned int) * DISTRICT_SIZE * 2);
        // cudaMemset(log->LOG_CUSTOMER_R, 0xffffffff, sizeof(unsigned int) * CUSTOMER_SIZE);
        // cudaMemset(log->LOG_CUSTOMER_W, 0xffffffff, sizeof(unsigned int) * CUSTOMER_SIZE);
        // cudaMemset(log->LOG_HISTORY_R, 0xffffffff, sizeof(unsigned int) * HISTORY_SIZE);
        // cudaMemset(log->LOG_HISTORY_W, 0xffffffff, sizeof(unsigned int) * HISTORY_SIZE);
        // cudaMemset(log->LOG_NEWORDER_R, 0xffffffff, sizeof(unsigned int) * NEWORDER_SIZE);
        // cudaMemset(log->LOG_NEWORDER_W, 0xffffffff, sizeof(unsigned int) * NEWORDER_SIZE);
        // cudaMemset(log->LOG_ORDER_R, 0xffffffff, sizeof(unsigned int) * ORDER_SIZE);
        // cudaMemset(log->LOG_ORDER_W, 0xffffffff, sizeof(unsigned int) * ORDER_SIZE);
        // cudaMemset(log->LOG_ORDERLINE_R, 0xffffffff, sizeof(unsigned int) * ORDERLINE_SIZE);
        // cudaMemset(log->LOG_ORDERLINE_W, 0xffffffff, sizeof(unsigned int) * ORDERLINE_SIZE);
        // cudaMemset(log->LOG_STOCK_R, 0xffffffff, sizeof(unsigned int) * STOCK_SIZE);
        // cudaMemset(log->LOG_STOCK_W, 0xffffffff, sizeof(unsigned int) * STOCK_SIZE);
        // cudaMemset(log->LOG_ITEM_R, 0xffffffff, sizeof(unsigned int) * ITEM_SIZE);
        // cudaMemset(log->LOG_ITEM_W, 0xffffffff, sizeof(unsigned int) * ITEM_SIZE);
        // cudaMemset(log->W_YTD, 0, sizeof(unsigned int) * PAYMENT_CNT);
        // cudaMemset(log->TMP_W_YTD, 0, sizeof(unsigned int) * WAREHOUSE_SIZE);
        // cudaMemset(log->D_YTD, 0, sizeof(unsigned int) * PAYMENT_CNT);
        // cudaMemset(log->TMP_D_YTD, 0, sizeof(unsigned int) * DISTRICT_SIZE);
    }

    void create_warehouse(unsigned int thID)
    {
        for (size_t i = 0; i < WAREHOUSE_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * WAREHOUSE_SIZE / WAREHOUSE_SIZE;
            warehouse_tbl[ii].W_ID = ii;
            warehouse_tbl[ii].W_TAX = 1;
            warehouse_tbl[ii].W_YTD = 1;
            warehouse_tbl[ii].W_NAME = 0;
            warehouse_tbl[ii].W_STREET_1 = 0;
            warehouse_tbl[ii].W_STREET_2 = 0;
            warehouse_tbl[ii].W_CITY = 10;
            warehouse_tbl[ii].W_STATE = 0;
            warehouse_tbl[ii].W_ZIP = 0;

            snapshot->warehouse_snapshot[0 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_ID;
            snapshot->warehouse_snapshot[1 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_TAX;
            snapshot->warehouse_snapshot[2 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_YTD;
            snapshot->warehouse_snapshot[3 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_NAME;
            snapshot->warehouse_snapshot[4 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_STREET_1;
            snapshot->warehouse_snapshot[5 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_STREET_2;
            snapshot->warehouse_snapshot[6 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_CITY;
            snapshot->warehouse_snapshot[7 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_STATE;
            snapshot->warehouse_snapshot[8 * WAREHOUSE_SIZE + ii] = warehouse_tbl[ii].W_ZIP;
        }
    }

    void create_district(unsigned int thID)
    {
        for (size_t i = 0; i < DISTRICT_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * DISTRICT_SIZE / WAREHOUSE_SIZE;
            district_tbl[ii].D_ID = ii % 10;
            district_tbl[ii].D_W_ID = ii / 10;
            district_tbl[ii].D_TAX = 1;
            district_tbl[ii].D_YTD = 1;
            district_tbl[ii].D_NEXT_O_ID = 0;
            district_tbl[ii].D_NAME = 0;
            district_tbl[ii].D_STREET_1 = 0;
            district_tbl[ii].D_STREET_2 = 0;
            district_tbl[ii].D_CITY = 0;
            district_tbl[ii].D_STATE = 0;
            district_tbl[ii].D_ZIP = 0;

            snapshot->district_snapshot[0 * DISTRICT_SIZE + ii] = district_tbl[ii].D_ID;
            snapshot->district_snapshot[1 * DISTRICT_SIZE + ii] = district_tbl[ii].D_W_ID;
            snapshot->district_snapshot[2 * DISTRICT_SIZE + ii] = district_tbl[ii].D_TAX;
            snapshot->district_snapshot[3 * DISTRICT_SIZE + ii] = district_tbl[ii].D_YTD;
            snapshot->district_snapshot[4 * DISTRICT_SIZE + ii] = district_tbl[ii].D_NEXT_O_ID;
            snapshot->district_snapshot[5 * DISTRICT_SIZE + ii] = district_tbl[ii].D_NAME;
            snapshot->district_snapshot[6 * DISTRICT_SIZE + ii] = district_tbl[ii].D_STREET_1;
            snapshot->district_snapshot[7 * DISTRICT_SIZE + ii] = district_tbl[ii].D_STREET_2;
            snapshot->district_snapshot[8 * DISTRICT_SIZE + ii] = district_tbl[ii].D_CITY;
            snapshot->district_snapshot[9 * DISTRICT_SIZE + ii] = district_tbl[ii].D_STATE;
            snapshot->district_snapshot[10 * DISTRICT_SIZE + ii] = district_tbl[ii].D_ZIP;
        }
    }

    void create_customer(unsigned int thID)
    {
        for (size_t i = 0; i < CUSTOMER_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * CUSTOMER_SIZE / WAREHOUSE_SIZE;
            customer_tbl[ii].C_ID = ii % 3000;
            customer_tbl[ii].C_W_ID = ii / 30000;
            customer_tbl[ii].C_D_ID = (ii / 3000) % 10;
            customer_tbl[ii].C_CREDIT_LIM = 100000;
            customer_tbl[ii].C_DISCOUNT = 100000;
            customer_tbl[ii].C_BALANCE = 100000;
            customer_tbl[ii].C_YTD_PAYMENT = 100000;
            customer_tbl[ii].C_PAYMENT_CNT = 100000;
            customer_tbl[ii].C_DELIVERY_CNT = 100000;
            customer_tbl[ii].C_DATA = 100000;
            customer_tbl[ii].C_CREDIT = random.uniform_dist(0, 1); // 0 bad; 1 good
            customer_tbl[ii].C_FIRST = 0;
            customer_tbl[ii].C_MIDDLE = 0;
            customer_tbl[ii].C_LAST = ii % 20; // random.uniform_dist(0, 15);
            customer_tbl[ii].C_STREET_1 = 0;
            customer_tbl[ii].C_STREET_2 = 0;
            customer_tbl[ii].C_CITY = 0;
            customer_tbl[ii].C_STATE = 0;
            customer_tbl[ii].C_ZIP = 0;
            customer_tbl[ii].C_PHONE = 0;
            customer_tbl[ii].C_SINCE = 0;
            customer_name_index[ii] = customer_tbl[ii].C_LAST;

            snapshot->customer_snapshot[0 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_ID;
            snapshot->customer_snapshot[1 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_W_ID;
            snapshot->customer_snapshot[2 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_D_ID;
            snapshot->customer_snapshot[3 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_CREDIT_LIM;
            snapshot->customer_snapshot[4 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_DISCOUNT;
            snapshot->customer_snapshot[5 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_BALANCE;
            snapshot->customer_snapshot[6 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_YTD_PAYMENT;
            snapshot->customer_snapshot[7 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_PAYMENT_CNT;
            snapshot->customer_snapshot[8 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_DELIVERY_CNT;
            snapshot->customer_snapshot[9 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_DATA;
            snapshot->customer_snapshot[10 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_CREDIT;
            snapshot->customer_snapshot[11 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_FIRST;
            snapshot->customer_snapshot[12 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_MIDDLE;
            snapshot->customer_snapshot[13 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_LAST;
            snapshot->customer_snapshot[14 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_STREET_1;
            snapshot->customer_snapshot[15 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_STREET_2;
            snapshot->customer_snapshot[16 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_CITY;
            snapshot->customer_snapshot[17 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_STATE;
            snapshot->customer_snapshot[18 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_ZIP;
            snapshot->customer_snapshot[19 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_PHONE;
            snapshot->customer_snapshot[20 * CUSTOMER_SIZE + ii] = customer_tbl[ii].C_SINCE;
            index->customer_name_index[ii] = customer_name_index[ii];
        }
    }
    void create_history(unsigned int thID)
    {
    }
    void create_neworder(unsigned int thID)
    {
    }
    void create_order(unsigned int thID)
    {
        for (size_t i = 0; i < ORDER_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * ORDER_SIZE / WAREHOUSE_SIZE;
            order_tbl[ii].O_ID = ii % 30000;
            order_tbl[ii].O_D_ID = ii / 3000;
            order_tbl[ii].O_W_ID = ii / 30000;
            order_tbl[ii].O_C_ID = random.uniform_dist(0, 2999);
            order_tbl[ii].O_ENTRY_D = 0;
            order_tbl[ii].O_CARRIER_ID = 0;
            order_tbl[ii].O_OL_CNT = 0;
            order_tbl[ii].O_ALL_LOCAL = 0;
            snapshot->order_snapshot[0 * ORDER_SIZE + ii] = order_tbl[ii].O_ID;
            snapshot->order_snapshot[1 * ORDER_SIZE + ii] = order_tbl[ii].O_D_ID;
            snapshot->order_snapshot[2 * ORDER_SIZE + ii] = order_tbl[ii].O_W_ID;
            snapshot->order_snapshot[3 * ORDER_SIZE + ii] = order_tbl[ii].O_C_ID;
            snapshot->order_snapshot[4 * ORDER_SIZE + ii] = order_tbl[ii].O_ENTRY_D;
            snapshot->order_snapshot[5 * ORDER_SIZE + ii] = order_tbl[ii].O_CARRIER_ID;
            snapshot->order_snapshot[6 * ORDER_SIZE + ii] = order_tbl[ii].O_OL_CNT;
            snapshot->order_snapshot[7 * ORDER_SIZE + ii] = order_tbl[ii].O_ALL_LOCAL;
        }
    }
    void create_orderline(unsigned int thID)
    {
        for (size_t i = 0; i < ORDERLINE_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * ORDERLINE_SIZE / WAREHOUSE_SIZE;
            orderline_tbl[ii].OL_O_ID = ii % 450000;
            orderline_tbl[ii].OL_D_ID = ii / 45000;
            orderline_tbl[ii].OL_W_ID = ii / 450000;
            orderline_tbl[ii].OL_NUMBER = 0;
            orderline_tbl[ii].OL_I_ID = random.uniform_dist(0, 99999);
            orderline_tbl[ii].OL_SUPPLY_W_ID = orderline_tbl[ii].OL_W_ID;
            orderline_tbl[ii].OL_DELIVERY_D = 0;
            orderline_tbl[ii].OL_QUANLITY = 0;
            orderline_tbl[ii].OL_AMOUNT = 0;
            orderline_tbl[ii].OL_DIST_INF = 0;
            snapshot->orderline_snapshot[0 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_O_ID;
            snapshot->orderline_snapshot[1 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_D_ID;
            snapshot->orderline_snapshot[2 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_W_ID;
            snapshot->orderline_snapshot[3 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_NUMBER;
            snapshot->orderline_snapshot[4 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_I_ID;
            snapshot->orderline_snapshot[5 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_SUPPLY_W_ID;
            snapshot->orderline_snapshot[6 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_DELIVERY_D;
            snapshot->orderline_snapshot[7 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_QUANLITY;
            snapshot->orderline_snapshot[8 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_AMOUNT;
            snapshot->orderline_snapshot[9 * ORDERLINE_SIZE + ii] = orderline_tbl[ii].OL_DIST_INF;
        }
    }
    void create_stock(unsigned int thID)
    {
        for (size_t i = 0; i < STOCK_SIZE / WAREHOUSE_SIZE; i++)
        {
            unsigned int ii = i + thID * STOCK_SIZE / WAREHOUSE_SIZE;
            stock_tbl[ii].S_I_ID = ii % 100000;
            stock_tbl[ii].S_W_ID = ii / 100000;
            stock_tbl[ii].S_QUANTITY = 100000;
            stock_tbl[ii].S_YTD = 100000;
            stock_tbl[ii].S_ORDER_CNT = 0;
            stock_tbl[ii].S_REMOVE_CNT = 0;
            stock_tbl[ii].S_DIST_01 = 1;
            stock_tbl[ii].S_DIST_02 = 2;
            stock_tbl[ii].S_DIST_03 = 3;
            stock_tbl[ii].S_DIST_04 = 4;
            stock_tbl[ii].S_DIST_05 = 5;
            stock_tbl[ii].S_DIST_06 = 6;
            stock_tbl[ii].S_DIST_07 = 7;
            stock_tbl[ii].S_DIST_08 = 8;
            stock_tbl[ii].S_DIST_09 = 9;
            stock_tbl[ii].S_DIST_10 = 10;
            stock_tbl[ii].S_DATA = 0;

            snapshot->stock_snapshot[0 * STOCK_SIZE + ii] = stock_tbl[ii].S_I_ID;
            snapshot->stock_snapshot[1 * STOCK_SIZE + ii] = stock_tbl[ii].S_W_ID;
            snapshot->stock_snapshot[2 * STOCK_SIZE + ii] = stock_tbl[ii].S_QUANTITY;
            snapshot->stock_snapshot[3 * STOCK_SIZE + ii] = stock_tbl[ii].S_YTD;
            snapshot->stock_snapshot[4 * STOCK_SIZE + ii] = stock_tbl[ii].S_ORDER_CNT;
            snapshot->stock_snapshot[5 * STOCK_SIZE + ii] = stock_tbl[ii].S_REMOVE_CNT;
            snapshot->stock_snapshot[6 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_01;
            snapshot->stock_snapshot[7 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_02;
            snapshot->stock_snapshot[8 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_03;
            snapshot->stock_snapshot[9 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_04;
            snapshot->stock_snapshot[10 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_05;
            snapshot->stock_snapshot[11 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_06;
            snapshot->stock_snapshot[12 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_07;
            snapshot->stock_snapshot[13 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_08;
            snapshot->stock_snapshot[14 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_09;
            snapshot->stock_snapshot[15 * STOCK_SIZE + ii] = stock_tbl[ii].S_DIST_10;
            snapshot->stock_snapshot[16 * STOCK_SIZE + ii] = stock_tbl[ii].S_DATA;
        }
    }
    void create_item()
    {
        for (size_t ii = 0; ii < ITEM_SIZE; ii++)
        {
            item_tbl[ii].I_ID = ii;
            item_tbl[ii].I_IM_ID = 1000;
            item_tbl[ii].I_PRICE = random.uniform_dist(1, 40000);
            item_tbl[ii].I_DATA = random.uniform_dist(1, 10);
            item_tbl[ii].I_NAME = ii;
            snapshot->item_snapshot[0 * ITEM_SIZE + ii] = item_tbl[ii].I_ID;
            snapshot->item_snapshot[1 * ITEM_SIZE + ii] = item_tbl[ii].I_IM_ID;
            snapshot->item_snapshot[2 * ITEM_SIZE + ii] = item_tbl[ii].I_PRICE;
            snapshot->item_snapshot[3 * ITEM_SIZE + ii] = item_tbl[ii].I_DATA;
            snapshot->item_snapshot[4 * ITEM_SIZE + ii] = item_tbl[ii].I_NAME;
        }
    }
    void print()
    {
        std::cout << "==========================================\n";
        std::cout << "\t WAREHOUSE_SIZE = " << WAREHOUSE_SIZE << std::endl;
        std::cout << "\t DISTRICT_SIZE  = " << DISTRICT_SIZE << std::endl;
        std::cout << "\t CUSTOMER_SIZE  = " << CUSTOMER_SIZE << std::endl;
        std::cout << "\t HISTORY_SIZE   = " << HISTORY_SIZE << std::endl;
        std::cout << "\t NEWORDER_SIZE  = " << NEWORDER_SIZE << std::endl;
        std::cout << "\t ORDER_SIZE     = " << ORDER_SIZE << std::endl;
        std::cout << "\t ORDERLINE_SIZE = " << ORDERLINE_SIZE << std::endl;
        std::cout << "\t STOCK_SIZE     = " << STOCK_SIZE << std::endl;
        std::cout << "\t ITEM_SIZE      = " << ITEM_SIZE << std::endl;
        std::cout << "------------------------------------------\n";
        std::cout << "\t TABLE_SIZE     = " << TABLE_SIZE_1D << std::endl;
        std::cout << "\t BATCH_SIZE     = " << BATCH_SIZE << std::endl;
        std::cout << "==========================================\n\n";
        std::cout << "SNAPSHOT is [" << (float)sizeof(SNAPSHOT) / 1024 / 1024 / 1024 << "GB].\n\n";
        std::cout << "INDEX is [" << (float)sizeof(INDEX) / 1024 / 1024 << "MB].\n\n";
        std::cout << "LOG is [" << (float)sizeof(LOG) / 1024 / 1024 / 1024 << "GB].\n\n";
    }
};
void initial_data(Database *database)
{
    long long start_t = current_time();
    // std::thread thread_warehouse(&Database::create_warehouse, database);
    // std::thread thread_district(&Database::create_district, database);
    // std::thread thread_customer(&Database::create_customer, database);
    // std::thread thread_history(&Database::create_history, database);
    // std::thread thread_neworder(&Database::create_neworder, database);
    // std::thread thread_order(&Database::create_order, database);
    // std::thread thread_orderline(&Database::create_orderline, database);
    // std::thread thread_stock(&Database::create_stock, database);
    // std::thread thread_item(&Database::create_item, database);
    // thread_warehouse.join();
    // thread_district.join();
    // thread_customer.join();
    // thread_history.join();
    // thread_neworder.join();
    // thread_order.join();
    // thread_orderline.join();
    // thread_stock.join();
    // thread_item.join();
    std::vector<std::thread> threads_warehouse;
    std::vector<std::thread> threads_district;
    std::vector<std::thread> threads_customer;
    std::vector<std::thread> threads_history;
    std::vector<std::thread> threads_neworder;
    std::vector<std::thread> threads_order;
    std::vector<std::thread> threads_orderline;
    std::vector<std::thread> threads_stock;

    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_warehouse.push_back(std::thread(&Database::create_warehouse, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_warehouse[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_district.push_back(std::thread(&Database::create_district, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_district[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_customer.push_back(std::thread(&Database::create_customer, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_customer[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_history.push_back(std::thread(&Database::create_history, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_history[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_neworder.push_back(std::thread(&Database::create_neworder, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_neworder[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_order.push_back(std::thread(&Database::create_order, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_order[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_orderline.push_back(std::thread(&Database::create_orderline, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_orderline[i].join();
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_stock.push_back(std::thread(&Database::create_stock, database, i));
    }
    for (size_t i = 0; i < WAREHOUSE_SIZE; i++)
    {
        threads_stock[i].join();
    }
    std::thread thread_item(std::thread(&Database::create_item, database));
    thread_item.join();
    cudaMemcpy(database->snapshot_d, database->snapshot, sizeof(SNAPSHOT), cudaMemcpyHostToDevice);
    cudaMemcpy(database->index_d, database->index, sizeof(INDEX), cudaMemcpyHostToDevice);
    long long end_t = current_time();
    database->time = duration(start_t, end_t);
    // database->print();
    std::cout << "Initiallization of all tables costs [" << database->time << "s].\n\n";
}
