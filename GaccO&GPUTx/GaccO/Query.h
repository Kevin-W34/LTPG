#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "Predefine.h"
#include "Datastructure.h"
#include "Genericfunction.h"
#include "Random.h"

class MakeNeworder
{
private:
    unsigned int *NOID;
    unsigned int *OID;
    unsigned int *OOLID;
    std::mutex *NOID_mut;
    std::mutex *OID_mut;
    std::mutex *OOLID_mut;
    Random random;

public:
    MakeNeworder()
    {
        NOID = new unsigned int[WAREHOUSE_SIZE];
        OID = new unsigned int[WAREHOUSE_SIZE];
        OOLID = new unsigned int[WAREHOUSE_SIZE];
        NOID_mut = new std::mutex[WAREHOUSE_SIZE];
        OID_mut = new std::mutex[WAREHOUSE_SIZE];
        OOLID_mut = new std::mutex[WAREHOUSE_SIZE];
        for (unsigned int i = 0; i < WAREHOUSE_SIZE; i++)
        {
            OID[i] = 0;
            OOLID[i] = 0;
            NOID[i] = 0;
        }
    }

    ~MakeNeworder()
    {
        free(NOID);
        free(OID);
        free(OOLID);
        free(NOID_mut);
        free(OID_mut);
        free(OOLID_mut);
    }

    void mutex_NOID(NeworderQuery *neworder)
    {
        this->NOID_mut[neworder->W_ID].lock();
        neworder->N_O_ID = this->NOID[neworder->W_ID];
        this->NOID[neworder->W_ID] += 1;
        if (this->NOID[neworder->W_ID] >= 30000)
        {
            this->NOID[neworder->W_ID] = 0;
        }
        this->NOID_mut[neworder->W_ID].unlock();
    }

    void mutex_OID(NeworderQuery *neworder)
    {
        this->OID_mut[neworder->W_ID].lock();
        neworder->O_ID = this->OID[neworder->W_ID];
        this->OID[neworder->W_ID] += 1;
        if (this->OID[neworder->W_ID] >= 30000)
        {
            this->OID[neworder->W_ID] = 0;
        }
        this->OID_mut[neworder->W_ID].unlock();
    }

    void mutex_OOLID(NeworderQuery *neworder)
    {
        this->OOLID_mut[neworder->W_ID].lock();
        neworder->O_OL_ID = this->OOLID[neworder->W_ID];
        this->OOLID[neworder->W_ID] += neworder->O_OL_CNT;
        if (this->OOLID[neworder->W_ID] >= 450000)
        {
            this->OOLID[neworder->W_ID] = 0;
        }
        this->OOLID_mut[neworder->W_ID].unlock();
    }

    NeworderQuery make()
    {
        NeworderQuery neworder;
        neworder.W_ID = (unsigned int)random.uniform_dist(0, WAREHOUSE_SIZE - 1); // [0 , WAREHOUSE_SIZE - 1]
        neworder.D_ID = (unsigned int)random.uniform_dist(0, 9);                  // [0 , 9]
        neworder.C_ID = (unsigned int)random.uniform_dist(0, 2999);

        neworder.O_OL_CNT = (unsigned int)random.uniform_dist(5, 15); // [5 , 15]
        // neworder.O_OL_CNT = 15;
        // neworder.O_OL_CNT = 1;
        mutex_NOID(&neworder);
        mutex_OID(&neworder);
        mutex_OOLID(&neworder);
        for (unsigned int i = 0; i < neworder.O_OL_CNT; i++)
        {
            // neworder.INFO[i].OL_I_ID = (unsigned int)random.non_uniform_distribution(8191, 1, 100000) - 1; //[0 , 99999];
            neworder.INFO[i].OL_I_ID = (unsigned int)random.uniform_dist(0, 99999);
            for (unsigned int k = 0; k < i; k++)
            {
                while (neworder.INFO[k].OL_I_ID == neworder.INFO[i].OL_I_ID)
                {
                    // neworder.INFO[i].OL_I_ID = (unsigned int)random.non_uniform_distribution(8191, 1, 100000) - 1;
                    neworder.INFO[i].OL_I_ID = (unsigned int)random.uniform_dist(0, 99999);
                }
            }
            neworder.INFO[i].OL_SUPPLY_W_ID = neworder.W_ID; // Home Warehouse
            // unsigned int isLocal = random.uniform_dist(1, 100);
            // if (isLocal == 1) // Remote Warehouse
            // {
            //     isLocal = random.uniform_dist(0, 7);
            //     while (isLocal == neworder.W_ID)
            //     {
            //         isLocal = random.uniform_dist(0, 7);
            //     }
            //     neworder.INFO[i].OL_SUPPLY_W_ID = isLocal;
            // }
            neworder.INFO[i].OL_QUANTITY = (unsigned int)random.uniform_dist(1, 2);
        }

        return neworder;
    } /* make new new order */
};

class MakePayment
{
private:
    unsigned int *HID;
    std::mutex *HID_mut;
    Random random;

public:
    MakePayment()
    {
        HID = new unsigned int[WAREHOUSE_SIZE];
        HID_mut = new std::mutex[WAREHOUSE_SIZE];
        for (unsigned int i = 0; i < WAREHOUSE_SIZE; i++)
        {
            HID[i] = 0;
        }
    }

    ~MakePayment()
    {
        free(HID);
        free(HID_mut);
    }

    void mutex_HID(PaymentQuery *payment)
    {
        this->HID_mut[payment->W_ID].lock();
        payment->H_ID = this->HID[payment->W_ID]++;
        if (this->HID[payment->W_ID] == 30000)
        {
            this->HID[payment->W_ID] = 0;
        }
        this->HID_mut[payment->W_ID].unlock();
    }

    PaymentQuery make()
    {
        PaymentQuery payment;
        payment.W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        // payment.W_ID = 0;
        payment.D_ID = random.uniform_dist(0, 9);
        // payment.D_ID = 1;
        unsigned int isName = random.uniform_dist(1, 100);
        if (isName < 61)
        {
            payment.isName = 1;
            payment.C_ID = 0; // random.non_uniform_distribution(1023, 1, 3000) - 1;
            payment.C_LAST = (random.uniform_dist(0, 19) << 8) + random.uniform_dist(0, 149);
        } /* C_LAST */
        else
        {
            payment.isName = 0;
            payment.C_ID = (unsigned int)random.uniform_dist(0, 2999);
            payment.C_LAST = 20;
        } /* C_ID */

        // unsigned int isLocal = random.uniform_dist(1, 100);
        // if (isLocal <= 85) // 85
        // {
        //     payment.C_W_ID = payment.W_ID;
        // } /* Local */
        // else
        // {
        //     payment.C_W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        //     while (payment.C_W_ID == payment.W_ID)
        //     {
        //         payment.C_W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        //     }
        // } /* Remote */

        mutex_HID(&payment);

        payment.C_W_ID = payment.W_ID;
        payment.C_D_ID = random.uniform_dist(0, 9);
        // payment.H_AMOUNT = random.uniform_dist(1, 5);
        payment.H_AMOUNT = 1;

        return payment;
    } /* make new payment */
};

class MakeOrderstatus
{
private:
    Random random;

public:
    MakeOrderstatus()
    {
    }
    ~MakeOrderstatus()
    {
    }
    OrderstatusQuery make()
    {
        OrderstatusQuery query;
        query.W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        query.D_ID = random.uniform_dist(0, 9);
        unsigned int isName = random.uniform_dist(1, 100);
        if (isName < 61)
        {
            query.isName = 1;
            query.C_ID = 0; // random.non_uniform_distribution(1023, 1, 3000) - 1;
            query.C_LAST = (random.uniform_dist(0, 19) << 8) + random.uniform_dist(0, 149);
        } /* C_LAST */
        else
        {
            query.isName = 0;
            query.C_ID = (unsigned int)random.uniform_dist(0, 2999);
            query.C_LAST = 20;
        } /* C_ID */
        return query;
    }
};
class MakeDelivery
{
private:
    Random random;

public:
    MakeDelivery()
    {
    }
    ~MakeDelivery()
    {
    }
    DeliveryQuery make()
    {
        DeliveryQuery query;
        for (size_t i = 0; i < 10; i++)
        {
            query.O_ID[i] = 0;
        }
        return query;
    }
};
class MakeStocklevel
{
private:
    Random random;

public:
    MakeStocklevel()
    {
    }
    ~MakeStocklevel()
    {
    }
    StockLevelQuery make()
    {
        StockLevelQuery query;
        query.W_ID = random.uniform_dist(0, WAREHOUSE_SIZE - 1);
        query.D_ID = random.uniform_dist(0, 9);
        query.query_cnt = random.uniform_dist(10, 20);
        return query;
    }
};

std::atomic<unsigned int> ID(0);
std::atomic<unsigned int> neworder_ID(0);
std::atomic<unsigned int> payment_ID(0);
std::atomic<unsigned int> analyse_n_ID(0);
std::atomic<unsigned int> analyse_p_ID(0);

class Query
{
private:
    Random random;

public:
    // NEWORDER_SET *neworder_set;
    // PAYMENT_SET *payment_set;
    NEWORDER_SET *neworder_set_new;
    PAYMENT_SET *payment_set_new;
    NEWORDER_SET *neworder_set_d;
    PAYMENT_SET *payment_set_d;
    NEWORDER_QUERY *neworder_query;
    PAYMENT_QUERY *payment_query;
    NEWORDER_QUERY *neworder_query_d;
    PAYMENT_QUERY *payment_query_d;
    MakeNeworder *makeNeworder;
    MakePayment *makePayment;
    NEWORDERQUERY_ACCESS *neworderquery_access_d;
    NEWORDERQUERY_AUXILIARY *neworderquery_auxiliary_d;
    PAYMENTQUERY_ACCESS *paymentquery_access_d;
    PAYMENTQUERY_AUXILIARY *paymentquery_auxiliary_d;
    unsigned int commit_neworder = 0;
    unsigned int commit_payment = 0;
    unsigned int commit_all = 0;
    unsigned int commit_n = 0;
    unsigned int commit_p = 0;
    float kernel_0_time_all = 0.0;
    float time_0_all = 0.0;
    float time_1_all = 0.0;
    float time_2_all = 0.0;
    float time_3_all = 0.0;
    float time_4_all = 0.0;
    float memory_speed = 0.0;

    Query()
    {
        // neworder_set = (NEWORDER_SET *)malloc(sizeof(NEWORDER_SET) * PRE_GEN_EPOCH);
        // payment_set = (PAYMENT_SET *)malloc(sizeof(PAYMENT_SET) * PRE_GEN_EPOCH);
        // neworder_set_new = (NEWORDER_SET *)malloc(sizeof(NEWORDER_SET));
        // payment_set_new = (PAYMENT_SET *)malloc(sizeof(PAYMENT_SET));
        // cudaMallocHost((void **)&neworder_set, sizeof(NEWORDER_SET) * PRE_GEN_EPOCH);
        // cudaMallocHost((void **)&payment_set, sizeof(PAYMENT_SET) * PRE_GEN_EPOCH);
        cudaMallocHost((void **)&neworder_query, sizeof(NEWORDER_QUERY) * PRE_GEN_EPOCH);
        cudaMallocHost((void **)&payment_query, sizeof(PAYMENT_QUERY) * PRE_GEN_EPOCH);
        cudaMallocHost((void **)&neworder_set_new, sizeof(NEWORDER_SET) * STREAM_SIZE);
        cudaMallocHost((void **)&payment_set_new, sizeof(PAYMENT_SET) * STREAM_SIZE);
        cudaMalloc((void **)&neworder_set_d, sizeof(NEWORDER_SET) * STREAM_SIZE);
        cudaMalloc((void **)&payment_set_d, sizeof(PAYMENT_SET) * STREAM_SIZE);
        cudaMalloc((void **)&neworder_query_d, sizeof(NEWORDER_QUERY) * STREAM_SIZE);
        cudaMalloc((void **)&payment_query_d, sizeof(PAYMENT_QUERY) * STREAM_SIZE);
        cudaMalloc((void **)&neworderquery_access_d, sizeof(NEWORDERQUERY_ACCESS));
        cudaMalloc((void **)&neworderquery_auxiliary_d, sizeof(NEWORDERQUERY_AUXILIARY));
        cudaMalloc((void **)&paymentquery_access_d, sizeof(PAYMENTQUERY_ACCESS));
        cudaMalloc((void **)&paymentquery_auxiliary_d, sizeof(PAYMENTQUERY_AUXILIARY));
        makeNeworder = new MakeNeworder();
        makePayment = new MakePayment();
    }
    ~Query()
    {
        // free(neworder_set);
        // free(payment_set);
        // free(neworder_set_new);
        // free(payment_set_new);
        // cudaFreeHost(neworder_set);
        // cudaFreeHost(payment_set);
        cudaFreeHost(neworder_query);
        cudaFreeHost(payment_query);
        cudaFreeHost(neworder_set_new);
        cudaFreeHost(payment_set_new);
        cudaFree(neworder_set_d);
        cudaFree(payment_set_d);
        cudaFree(neworder_query_d);
        cudaFree(payment_query_d);
        cudaFree(neworderquery_access_d);
        cudaFree(neworderquery_auxiliary_d);
        cudaFree(paymentquery_access_d);
        cudaFree(paymentquery_auxiliary_d);
        free(makeNeworder);
        free(makePayment);
    }
    void clear_QUERY()
    {
        // memset(neworder_set, 0, sizeof(NEWORDER_SET) * PRE_GEN_EPOCH);
        // memset(payment_set, 0, sizeof(PAYMENT_SET) * PRE_GEN_EPOCH);
        memset(neworder_query, 0, sizeof(NEWORDER_QUERY) * PRE_GEN_EPOCH);
        memset(payment_query, 0, sizeof(PAYMENT_QUERY) * PRE_GEN_EPOCH);
    }
    void clear_auxliary()
    {
        cudaMemset(neworderquery_access_d, 0, sizeof(NEWORDERQUERY_ACCESS));
        cudaMemset(neworderquery_auxiliary_d, 0, sizeof(NEWORDERQUERY_AUXILIARY));
        cudaMemset(paymentquery_access_d, 0, sizeof(PAYMENTQUERY_ACCESS));
        cudaMemset(paymentquery_auxiliary_d, 0, sizeof(PAYMENTQUERY_AUXILIARY));
    }
    void statistic(unsigned int BATCH_ID)
    {
        // copy_to_host();
        unsigned int n_ID = 0;
        unsigned int p_ID = 0;
        for (size_t i = 0; i < NEWORDER_CNT / CPU_THREAD_SIZE; i++)
        {
            // if (neworder_set_new->COMMIT_AND_ABORT[i] == 1)
            n_ID = analyse_n_ID++;
            if (neworder_set_new->COMMIT_AND_ABORT[n_ID] == 1)
            {
                commit_n += 1;
            }
            // std::cout << "neworder " << i << " raw = " << neworder_set_new->raw[i] << " war = " << neworder_set_new->war[i] << " waw = " << neworder_set_new->waw[i] << std::endl;
            // std::cout << "neworder " << i << " raw = " << neworder_set->raw[i] << " war = " << neworder_set->war[i] << " waw = " << neworder_set->waw[i] << std::endl;
        }
        for (size_t i = 0; i < PAYMENT_CNT / CPU_THREAD_SIZE; i++)
        {
            // if (payment_set_new->COMMIT_AND_ABORT[i] == 1)
            p_ID = analyse_p_ID++;
            if (payment_set_new->COMMIT_AND_ABORT[p_ID] == 1)
            {
                commit_p += 1;
            }
            // std::cout << "payment " << i << " raw = " << payment_set_new->raw[i] << " war = " << payment_set_new->war[i] << " waw = " << payment_set_new->waw[i] << std::endl;
            // std::cout << "payment " << i << " raw = " << payment_set->raw[i] << " war = " << payment_set->war[i] << " waw = " << payment_set->waw[i] << std::endl;
        }
        // std::cout << "commit " << commit_n << " neworder\n";
        // commit_neworder += commit_n;
        // std::cout << "commit " << commit_p << " payment\n";
        // commit_payment += commit_p;
        // commit_all += commit_n + commit_p;
    }
    void random_query(unsigned int BATCH_ID)
    {
        for (unsigned int i = 0; i < BATCH_SIZE / CPU_THREAD_SIZE;)
        {
            unsigned int type = random.uniform_dist(1, 100);
            if (type < 51)
            {
                // std::cout << ID++ << " n " << neworder_ID++ << std::endl;
                if (neworder_ID >= NEWORDER_CNT)
                {
                    continue;
                }
                unsigned int q_id = neworder_ID++;
                unsigned int tid = ID++;
                NeworderQuery query = makeNeworder->make();
                // neworder_set->query[q_id] = makeNeworder->make();
                neworder_query[BATCH_ID].W_ID[q_id] = query.W_ID;
                neworder_query[BATCH_ID].D_ID[q_id] = query.D_ID;
                neworder_query[BATCH_ID].C_ID[q_id] = query.C_ID;
                neworder_query[BATCH_ID].O_ID[q_id] = query.O_ID;
                neworder_query[BATCH_ID].N_O_ID[q_id] = query.N_O_ID;
                neworder_query[BATCH_ID].O_OL_CNT[q_id] = query.O_OL_CNT;
                neworder_query[BATCH_ID].O_OL_ID[q_id] = query.O_OL_ID;
                for (size_t ii = 0; ii < 15; ii++)
                {
                    neworder_query[BATCH_ID].OL_I_ID[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_I_ID;
                    neworder_query[BATCH_ID].OL_SUPPLY_W_ID[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_SUPPLY_W_ID;
                    neworder_query[BATCH_ID].OL_QUANTITY[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_QUANTITY;
                }
                // neworder_set[BATCH_ID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
                neworder_query[BATCH_ID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
                i++;
                // std::cout << "n" << ((BATCH_ID & 0xfff) << 20) + tid << std::endl;
            }
            else
            {
                // std::cout << ID++ << " p " << payment_ID++ << std::endl;
                if (payment_ID >= PAYMENT_CNT)
                {
                    continue;
                }
                unsigned int q_id = payment_ID++;
                unsigned int tid = ID++;
                PaymentQuery query = makePayment->make();
                // payment_set->query[q_id] = makePayment->make();
                payment_query[BATCH_ID].W_ID[q_id] = query.W_ID;
                payment_query[BATCH_ID].D_ID[q_id] = query.D_ID;
                payment_query[BATCH_ID].C_ID[q_id] = query.C_ID;
                payment_query[BATCH_ID].C_LAST[q_id] = query.C_LAST;
                payment_query[BATCH_ID].isName[q_id] = query.isName;
                payment_query[BATCH_ID].C_D_ID[q_id] = query.C_D_ID;
                payment_query[BATCH_ID].C_W_ID[q_id] = query.C_W_ID;
                payment_query[BATCH_ID].H_AMOUNT[q_id] = query.H_AMOUNT;
                payment_query[BATCH_ID].H_ID[q_id] = query.H_ID;
                // payment_set[BATCH_ID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
                payment_query[BATCH_ID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
                i++;
                // std::cout << "p" << ((BATCH_ID & 0xfff) << 20) + tid << std::endl;
            }
        }
    }
    void initial_neworder(unsigned int epochID, unsigned int BATCH_ID)
    {
        for (unsigned int i = 0; i < NEWORDER_CNT / CPU_THREAD_SIZE;)
        {
            unsigned int q_id = neworder_ID++;
            unsigned int tid = ID++;
            NeworderQuery query = makeNeworder->make();
            // neworder_set->query[q_id] = makeNeworder->make();
            neworder_query[epochID].W_ID[q_id] = query.W_ID;
            neworder_query[epochID].D_ID[q_id] = query.D_ID;
            neworder_query[epochID].C_ID[q_id] = query.C_ID;
            neworder_query[epochID].O_ID[q_id] = query.O_ID;
            neworder_query[epochID].N_O_ID[q_id] = query.N_O_ID;
            neworder_query[epochID].O_OL_CNT[q_id] = query.O_OL_CNT;
            neworder_query[epochID].O_OL_ID[q_id] = query.O_OL_ID;
            for (size_t ii = 0; ii < 15; ii++)
            {
                neworder_query[epochID].OL_I_ID[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_I_ID;
                neworder_query[epochID].OL_SUPPLY_W_ID[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_SUPPLY_W_ID;
                neworder_query[epochID].OL_QUANTITY[NEWORDER_CNT * ii + q_id] = query.INFO[ii].OL_QUANTITY;
            }
            // neworder_set[epochID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
            neworder_query[epochID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
            i++;
        }
    }
    void initial_payment(unsigned int epochID, unsigned int BATCH_ID)
    {
        for (unsigned int i = 0; i < PAYMENT_CNT / CPU_THREAD_SIZE;)
        {
            unsigned int q_id = payment_ID++;
            unsigned int tid = ID++;
            PaymentQuery query = makePayment->make();
            // payment_set->query[q_id] = makePayment->make();
            payment_query[epochID].W_ID[q_id] = query.W_ID;
            payment_query[epochID].D_ID[q_id] = query.D_ID;
            payment_query[epochID].C_ID[q_id] = query.C_ID;
            payment_query[epochID].C_LAST[q_id] = query.C_LAST;
            payment_query[epochID].isName[q_id] = query.isName;
            payment_query[epochID].C_D_ID[q_id] = query.C_D_ID;
            payment_query[epochID].C_W_ID[q_id] = query.C_W_ID;
            payment_query[epochID].H_AMOUNT[q_id] = query.H_AMOUNT;
            payment_query[epochID].H_ID[q_id] = query.H_ID;
            // payment_set[epochID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
            payment_query[epochID].TID[q_id] = ((BATCH_ID & 0xfff) << 20) + tid;
            i++;
        }
    }
    void random_choose_query(unsigned int epochID, cudaStream_t *stream)
    {
        unsigned int choose = random.uniform_dist(0, PRE_GEN_EPOCH - 1);
        copy_to_device(choose, epochID, stream);
    }
    void copy_to_device(unsigned int choose, unsigned int epochID, cudaStream_t *stream)
    {
        long long start = current_time();
        cudaMemset(&neworder_set_d[epochID % STREAM_SIZE], 0, sizeof(NEWORDER_SET));
        cudaMemset(&payment_set_d[epochID % STREAM_SIZE], 0, sizeof(PAYMENT_SET));
        // cudaMemcpy(&neworder_set_d[epochID % 2], &neworder_set[choose], sizeof(NEWORDER_SET), cudaMemcpyHostToDevice);
        // cudaMemcpy(&payment_set_d[epochID % 2], &payment_set[choose], sizeof(PAYMENT_SET), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(&neworder_query_d[epochID % STREAM_SIZE], &neworder_query[choose], sizeof(NEWORDER_QUERY), cudaMemcpyHostToDevice, stream[epochID % STREAM_SIZE]);
        cudaMemcpyAsync(&payment_query_d[epochID % STREAM_SIZE], &payment_query[choose], sizeof(PAYMENT_QUERY), cudaMemcpyHostToDevice, stream[epochID % STREAM_SIZE]);
        cudaStreamSynchronize(stream[epochID % STREAM_SIZE]);
        long long end = current_time();
        float time = duration(start, end);
        std::cout << "copy to device costs [" << time << "s].\n";
        float query_size = (sizeof(NEWORDER_QUERY) + sizeof(PAYMENT_QUERY)) / 1024;
        // std::cout << "query_aize is [" << query_size << " KB].\n";
        float speed = query_size / time / 1024 / 1024;
        if (epochID >= WARMUP_TP && epochID < EPOCH_TP - WARMUP_TP)
        {
            this->memory_speed += speed;
        }
        // std::cout << "query_aize speed is [" << speed << " GB/s].\n";
    }
    void copy_to_host(unsigned int epochID, cudaStream_t *stream)
    {
        long long start = current_time();
        cudaMemcpyAsync(&neworder_set_new[epochID % STREAM_SIZE].COMMIT, &neworder_set_d[epochID % STREAM_SIZE].COMMIT, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[epochID % STREAM_SIZE]);
        cudaMemcpyAsync(&payment_set_new[epochID % STREAM_SIZE].COMMIT, &payment_set_d[epochID % STREAM_SIZE].COMMIT, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[epochID % STREAM_SIZE]);
        cudaStreamSynchronize(stream[epochID % STREAM_SIZE]);
        long long end = current_time();
        float time = duration(start, end);
        std::cout << "copy to host costs [" << time << " s].\n";
    }
    void print()
    {
        std::cout << "NEWORDER_SET is [" << (float)sizeof(NEWORDER_SET) / 1024 / 1024 << " MB].\n\n";
        std::cout << "PAYMENT_SET is [" << (float)sizeof(PAYMENT_SET) / 1024 / 1024 << " MB].\n\n";
        std::cout << "NEWORDER QUERY is [" << (float)sizeof(NEWORDER_QUERY) / 1024 / 1024 << " MB].\n\n";
        std::cout << "PAYMENT QUERY is [" << (float)sizeof(PAYMENT_QUERY) / 1024 / 1024 << " MB].\n\n";
    }
};
void initial_new_query(Query *query)
{
    query->clear_QUERY();
    long long start_t = current_time();
    for (size_t epoch_ID = 0; epoch_ID < PRE_GEN_EPOCH; epoch_ID++)
    {
        std::vector<std::thread> threads_n;
        std::vector<std::thread> threads_p;
        for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
        {
            threads_n.push_back(std::thread(&Query::initial_neworder, query, epoch_ID, i));
            threads_p.push_back(std::thread(&Query::initial_payment, query, epoch_ID, i));
        }
        for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
        {
            threads_n[i].join();
            threads_p[i].join();
        }
        neworder_ID = 0;
        payment_ID = 0;
        ID = 0;
    }
    // for (unsigned int i = 0; i < CPU_THREAD_SIZE * PRE_GEN_EPOCH; i++)
    // {
    //     threads_n[i].join();
    //     threads_p[i].join();
    // }
    long long end_t = current_time();
    float time = duration(start_t, end_t);
    std::cout << "Initiallization of " << PRE_GEN_EPOCH << " queries costs [" << time << " s].\n";
}
void make_new_query(unsigned int BATCH_ID, Query *query)
{
    query->clear_QUERY();
    long long start_t = current_time();
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
    {
        threads.push_back(std::thread(&Query::random_query, query, BATCH_ID));
    }
    for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
    {
        threads[i].join();
    }
    neworder_ID = 0;
    payment_ID = 0;
    long long end_t = current_time();
    float time = duration(start_t, end_t);
    // if (BATCH_ID == 0)
    // {
    //     query->print();
    // }

    // query->copy_to_device();
    // std::cout << "Initiallization of queries costs [" << time << "s].\n";
}
void statistic_query(unsigned int epochID, Query *query, cudaStream_t *stream)
{
    long long start_t = current_time();
    // std::vector<std::thread> threads;
    // for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
    // {
    //     threads.push_back(std::thread(&Query::statistic, query, BATCH_ID));
    // }
    // for (unsigned int i = 0; i < CPU_THREAD_SIZE; i++)
    // {
    //     threads[i].join();
    // }
    long long end_t = current_time();
    float time = duration(start_t, end_t);
    // analyse_n_ID = 0;
    // analyse_p_ID = 0;
    query->copy_to_host(epochID, stream);
    if (epochID >= WARMUP_TP && epochID < EPOCH_TP - WARMUP_TP)
    {
        // query->commit_neworder += query->neworder_set_new[epochID % STREAM_SIZE].COMMIT;
        // query->commit_payment += query->payment_set_new[epochID % STREAM_SIZE].COMMIT;
        query->commit_neworder += NEWORDER_CNT;
        query->commit_payment += PAYMENT_CNT;
        // std::cout << "commit neworder " << query->commit_n << std::endl;
        // std::cout << "commit payment " << query->commit_p << std::endl;
        query->commit_all = query->commit_neworder + query->commit_payment;
        query->commit_n = 0;
        query->commit_p = 0;
    }
    // std::cout << "Analyse of queries costs [" << time << "s].\n";
}
