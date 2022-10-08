#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Database.h"
#include "Query.h"
#include "Random.h"

struct State
{
    int64_t TID;
    int type;
    int lock = 1;
    int rank = 0;
    int war = 0;
    int raw = 0;
    int waw = 0;
};

struct SET
{
    int op_type; // 1 write
    int64_t TID;
    int32_t Loc;
    GLOBAL local_set;
};

struct Set_n
{
    int64_t TID;
    NewOrderQuery query;
    int lock = 1;
    int rank = 0;
    int war = 0;
    int raw = 0;
    int waw = 0;
};

struct Set_p
{
    int64_t TID;
    Payment query;
    int lock = 1;
    int rank = 0;
    int war = 0;
    int raw = 0;
    int waw = 0;
};

class Transaction
{
public:
    Set_n *neworder_vec;
    Set_n *neworder_d;
    Set_n *neworder_new;

    Set_p *payment_vec;
    Set_p *payment_d;
    Set_p *payment_new;

    std::vector<State> state_vec;
    State *state_d;

    std::vector<SET> set_vec;
    SET *set_d;

    std::vector<int> set_loc_vec;
    int *set_loc_list;
    int *set_loc_d;

    std::vector<int> set_TID_vec;
    int *set_TID_list;
    int *set_TID_d;

    std::vector<int> barrier_loc_vec;
    int *barrier_loc_d;
    std::vector<int> barrier_length_vec;
    int *barrier_length_d;
    int *rank_in_barrier_d;
    int *rank_in_barrier_new;

    GLOBAL *current_2D;
    GLOBAL *current_2D_d;
    Database database;
    MakeNewOrder n_make;
    MakePayment p_make;
    Random random;

    int32_t TID_global = 0;
    int OPID_global = 0;
    int TABLE_SIZE = TABLE_SIZE_1D;
    int n_defined_cnt = BATCH_SIZE * NEWORDERPERCENT / 100;
    int p_defined_cnt = BATCH_SIZE - n_defined_cnt;
    int n_remain_cnt = 0;
    int p_remain_cnt = 0;
    int all_op_cnt = 0;
    int th_cnt;
    int n_commit;
    int p_commit;
    int all_commit = 0;
    int n_commit_all = 0;
    int p_commit_all = 0;
    int rank_largest = 0;

    void random_new_query(int NO)
    {
        std::cout << "\nstart make new\n";
        OPID_global = 0;
        TID_global = 0;
        if (NO == 0)
        {
            neworder_vec = new Set_n[n_defined_cnt];
            payment_vec = new Set_p[p_defined_cnt];
        }
        n_remain_cnt = 0;
        p_remain_cnt = 0;
        int nsize = n_remain_cnt;
        int psize = p_remain_cnt;
        for (int i = 0; i < BATCH_SIZE - n_remain_cnt - p_remain_cnt;)
        {
            int probability = random.uniform_dist(1, 100);
            if (probability <= NEWORDERPERCENT)
            {
                if (nsize >= n_defined_cnt)
                {
                    continue;
                }
                Transaction::new_neworder(nsize);
                nsize += 1;
                i++;
            }
            else
            {
                if (psize >= p_defined_cnt)
                {
                    continue;
                }
                Transaction::new_payment(psize);
                psize += 1;
                i++;
            }
        }
        std::cout << nsize << std::endl;
        std::cout << psize << std::endl;
        std::cout << "end make new\n\n";
        // for (int i = 0; i < neworder_vec.size(); i++)
        // {
        //     std::cout << "neworder_vec[" << i << "].TID = " << neworder_vec[i].TID << std::endl;
        // }
        // for (int i = 0; i < payment_vec.size(); i++)
        // {
        //     std::cout << "payment_vec[" << i << "].TID = " << payment_vec[i].TID << std::endl;
        // }
    }
    void new_neworder(int Loc)
    {
        NewOrderQuery query = n_make.make();
        Set_n newquery;
        newquery.query = query;
        newquery.TID = TID_global;
        neworder_vec[Loc] = newquery;
        { // warehouse
            SET set;
            set.op_type = 0;
            set.Loc = query.W_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // district
            SET set;
            set.op_type = 0;
            set.Loc = WAREHOUSE_SIZE + query.D_ID + query.W_ID * 10;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // customer
            SET set;
            set.op_type = 0;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + query.W_ID * 30000 + 3000 * query.D_ID + query.C_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // neworder
            SET set;
            set.op_type = 1;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + query.W_ID * 30000 + query.N_O_ID;
            set.local_set.data[0] = query.N_O_ID;
            set.local_set.data[1] = query.D_ID;
            set.local_set.data[2] = query.W_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // order
            SET set;
            set.op_type = 1;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + query.W_ID * 30000 + query.O_ID;
            set.local_set.data[0] = query.O_ID;
            set.local_set.data[1] = query.D_ID;
            set.local_set.data[2] = query.W_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        for (size_t i = 0; i < query.O_OL_CNT; i++)
        {
            { // item
                SET set;
                set.op_type = 0;
                set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + STOCK_SIZE + query.INFO[i].OL_I_ID;
                set.TID = TID_global;
                set_vec.push_back(set);
                set_loc_vec.push_back(set.Loc);
                set_TID_vec.push_back(set.TID);
                OPID_global += 1;
            }
            { // stock
                SET set;
                set.op_type = 1;
                set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + ORDERLINE_SIZE + (query.INFO[i].OL_SUPPLY_W_ID * 100000) + query.INFO[i].OL_I_ID;
                ;
                set.local_set.data[2] = -query.INFO[i].OL_QUANTITY;
                set.local_set.data[3] = query.INFO[i].OL_QUANTITY * 2;
                set.local_set.data[4] = query.INFO[i].OL_QUANTITY;
                set.TID = TID_global;
                set_vec.push_back(set);
                set_loc_vec.push_back(set.Loc);
                set_TID_vec.push_back(set.TID);
                OPID_global += 1;
            }
            { // orderline
                SET set;
                set.op_type = 1;
                set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + HISTORY_SIZE + NEWORDER_SIZE + ORDER_SIZE + query.W_ID * 450000 + query.O_OL_ID + i;
                set.local_set.data[0] = query.O_OL_ID + i;
                set.local_set.data[1] = query.O_ID;
                set.local_set.data[2] = query.W_ID;
                set.local_set.data[3] = 0;
                set.local_set.data[4] = query.INFO[i].OL_I_ID;
                set.local_set.data[5] = query.INFO[i].OL_SUPPLY_W_ID;
                set.local_set.data[6] = 0;
                set.local_set.data[7] = query.INFO[i].OL_QUANTITY;
                set.local_set.data[8] = query.INFO[i].OL_QUANTITY * 2;
                set.local_set.data[9] = 0;
                set.TID = TID_global;
                set_vec.push_back(set);
                set_loc_vec.push_back(set.Loc);
                set_TID_vec.push_back(set.TID);
                OPID_global += 1;
            }
        }
        State state;
        state.TID = TID_global;
        state.type = NEWORDER_TYPE;
        state_vec.push_back(state);
        TID_global += 1;
    }
    void new_payment(int Loc)
    {
        Payment query = p_make.make();
        Set_p newquery;
        newquery.query = query;
        newquery.TID = TID_global;
        payment_vec[Loc] = newquery;
        { // warehouse
            SET set;
            set.op_type = 1;
            set.Loc = query.W_ID;
            set.local_set.data[2] = query.H_AMOUNT;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // warehouse
            SET set;
            set.op_type = 0;
            set.Loc = query.W_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // district
            SET set;
            set.op_type = 1;
            set.Loc = WAREHOUSE_SIZE + query.W_ID * 10 + query.D_ID;

            set.local_set.data[3] = query.H_AMOUNT;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // district
            SET set;
            set.op_type = 0;
            set.Loc = WAREHOUSE_SIZE + query.W_ID * 10 + query.D_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // customer
            if (query.isName)
            {
                int clast = query.C_LAST;
                for (size_t j = 0; j < 3000; j++)
                {
                    if (current_2D[WAREHOUSE_SIZE + DISTRICT_SIZE + query.C_W_ID * 30000 + query.C_D_ID * 3000 + j].data[13] == clast)
                    {
                        query.C_ID = current_2D[WAREHOUSE_SIZE + DISTRICT_SIZE + query.C_W_ID * 30000 + query.C_D_ID * 3000 + j].data[0];
                        break;
                    }
                }
            }
            SET set;
            set.op_type = 0;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + query.C_W_ID * 30000 + query.C_D_ID * 3000 + query.C_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // customer
            SET set;
            set.op_type = 1;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + query.C_W_ID * 30000 + query.C_D_ID * 3000 + query.C_ID;
            if (current_2D[WAREHOUSE_SIZE + DISTRICT_SIZE + query.C_W_ID * 30000 + query.C_D_ID * 3000 + query.C_ID].data[10])
            {
                set.local_set.data[5] = query.H_AMOUNT;
            }
            else
            {
                set.local_set.data[9] = query.H_AMOUNT;
            }
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        { // history
            SET set;
            set.op_type = 1;
            set.Loc = WAREHOUSE_SIZE + DISTRICT_SIZE + CUSTOMER_SIZE + query.W_ID * 30000 + query.H_ID;
            set.local_set.data[0] = query.C_ID;
            set.local_set.data[1] = query.C_D_ID;
            set.local_set.data[2] = query.C_W_ID;
            set.local_set.data[3] = query.D_ID;
            set.local_set.data[4] = query.W_ID;
            set.TID = TID_global;
            set_vec.push_back(set);
            set_loc_vec.push_back(set.Loc);
            set_TID_vec.push_back(set.TID);
            OPID_global += 1;
        }
        State state;
        state.TID = TID_global;
        state.type = PAYMENT_TYPE;
        state_vec.push_back(state);
        TID_global += 1;
    }

    void copy_from_host_to_device_data()
    {
        std::cout << "\nstart copy data from host to device\n";
        time_t start_cpy_data = clock();
        current_2D = database.getSnapShot_2D();
        cudaMalloc((void **)&current_2D_d, sizeof(GLOBAL) * TABLE_SIZE);
        cudaMemcpy(current_2D_d, current_2D, sizeof(GLOBAL) * TABLE_SIZE, cudaMemcpyHostToDevice);
        time_t end_cpy_data = clock();
        std::cout << "copy data cost = " << (float)(end_cpy_data - start_cpy_data) / 1000000.0 << std::endl;
        std::cout << "end copy data from host to device\n\n";
    }

    void copy_from_host_to_device_tx()
    {
        std::cout << "\nstart copy tx from host to device\n";
        cudaMalloc((void **)&neworder_d, sizeof(Set_n) * n_defined_cnt);
        cudaMalloc((void **)&payment_d, sizeof(Set_p) * p_defined_cnt);
        cudaMemcpy(neworder_d, neworder_vec, sizeof(Set_n) * n_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(payment_d, payment_vec, sizeof(Set_p) * p_defined_cnt, cudaMemcpyHostToDevice);
        std::cout << "end copy tx from host to device\n\n";
    }

    void copy_from_device_to_host_tx(int NO)
    {
        std::cout << "\nstart copy tx from device to host\n";
        neworder_new = new Set_n[n_defined_cnt];
        payment_new = new Set_p[p_defined_cnt];
        cudaMemcpy(neworder_new, neworder_d, sizeof(Set_n) * n_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(payment_new, payment_d, sizeof(Set_p) * p_defined_cnt, cudaMemcpyDeviceToHost);
        int pre_neworder_size = n_defined_cnt;
        int pre_payment_size = p_defined_cnt;
        free(neworder_vec);
        free(payment_vec);
        neworder_vec = new Set_n[n_defined_cnt];
        payment_vec = new Set_p[p_defined_cnt];
        n_commit = 0;
        int n_abort = 0;
        p_commit = 0;
        int p_abort = 0;
        n_remain_cnt = 0;
        p_remain_cnt = 0;
        for (size_t i = 0; i < pre_neworder_size; i++)
        {
            int state = neworder_new[i].waw || (neworder_new[i].war && neworder_new[i].raw);
            if (state > 0)
            {
                n_abort += 1;
            }
            else
            {
                n_commit += 1;
            }
        }

        for (size_t i = 0; i < pre_payment_size; i++)
        {
            int state = payment_new[i].waw || (payment_new[i].war && payment_new[i].raw);
            if (state > 0)
            {
                p_abort += 1;
            }
            else
            {
                p_commit += 1;
            }
        }
        if (NO > WARMUP - 1 && NO < EPOCH - WARMUP)
        {
            all_commit += p_commit;
            all_commit += n_commit;
            n_commit_all += n_commit;
            p_commit_all += p_commit;
        }
        cudaFree(neworder_d);
        cudaFree(payment_d);
        free(neworder_new);
        free(payment_new);

        std::cout << "abort neworder = " << n_abort << std::endl;
        std::cout << "abort payment = " << p_abort << std::endl;
        std::cout << "commit neworder = " << n_commit << std::endl;
        std::cout << "commit payment = " << p_commit << std::endl;
        std::cout << "end copy tx from device to host\n\n";
    }

    void close_gpu()
    {
        // free(neworder_vec);
        // free(payment_vec);

        cudaFree(current_2D_d);
        free(current_2D);
    }

    void reset_gpu()
    {
        cudaFree(set_loc_d);
        cudaFree(set_TID_d);
        cudaFree(barrier_loc_d);
        cudaFree(barrier_length_d);
        cudaFree(state_d);
        cudaFree(set_d);
    }

    void sort_v_and_id()
    {
        set_loc_list = new int[set_loc_vec.size()];
        set_TID_list = new int[set_TID_vec.size()];
        // thrust::device_vector<int> set_loc_d(set_loc_vec.begin(), set_loc_vec.end());
        // thrust::device_vector<int> set_TID_d(set_TID_vec.begin(), set_TID_vec.end());
        for (size_t i = 0; i < set_loc_vec.size(); i++)
        {
            set_loc_list[i] = set_loc_vec[i];
            set_TID_list[i] = set_TID_vec[i];
        }

        thrust::sort_by_key(set_loc_list, set_loc_list + set_loc_vec.size(), set_TID_list);
        int barrier = 0;

        for (size_t i = 1; i < set_loc_vec.size(); i++)
        {
            if (set_loc_list[i] != set_loc_list[barrier])
            {
                thrust::sort_by_key(set_TID_list + barrier, set_TID_list + i - 1, set_loc_list + barrier);
                barrier_loc_vec.push_back(barrier);
                barrier_length_vec.push_back(i - barrier);
                barrier = i;
            }
        }
        cudaMalloc((void **)&set_loc_d, sizeof(int) * set_loc_vec.size());
        cudaMalloc((void **)&set_TID_d, sizeof(int) * set_TID_vec.size());
        cudaMalloc((void **)&barrier_loc_d, sizeof(int) * barrier_loc_vec.size());
        cudaMalloc((void **)&barrier_length_d, sizeof(int) * barrier_length_vec.size());
        cudaMalloc((void **)&rank_in_barrier_d, sizeof(int) * set_TID_vec.size());
        cudaMalloc((void **)&state_d, sizeof(State) * state_vec.size());
        cudaMalloc((void **)&set_d, sizeof(SET) * set_vec.size());
        cudaMemcpy(set_loc_d, set_loc_list, sizeof(int) * set_loc_vec.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(set_TID_d, set_TID_list, sizeof(int) * set_TID_vec.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(barrier_loc_d, &(barrier_loc_vec[0]), sizeof(int) * barrier_loc_vec.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(barrier_length_d, &(barrier_length_vec[0]), sizeof(int) * barrier_length_vec.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(state_d, &(state_vec[0]), sizeof(State) * state_vec.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(set_d, &(set_vec[0]), sizeof(SET) * set_vec.size(), cudaMemcpyHostToDevice);

        set_loc_vec.clear();
        set_TID_vec.clear();
        state_vec.clear();
        set_vec.clear();
        free(set_loc_list);
        free(set_TID_list);
        barrier_loc_vec.clear();
        barrier_length_vec.clear();
    }

    void copy_rank_device_to_host()
    {
        rank_in_barrier_new = new int[set_TID_vec.size()];
        cudaMemcpy(rank_in_barrier_new, rank_in_barrier_d, sizeof(int) * set_TID_vec.size(), cudaMemcpyDeviceToHost);
        rank_largest = 0;
        for (size_t i = 0; i < set_TID_vec.size(); i++)
        {
            if (rank_largest < rank_in_barrier_new[i])
            {
                rank_largest = rank_in_barrier_new[i];
            }
        }
        free(rank_in_barrier_new);
        cudaFree(rank_in_barrier_d);
    }

    __device__ void step_1(int size) // barrier_length_d size
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (thID < size)
        {
            int length = barrier_length_d[thID];
            int start = barrier_loc_d[thID];
            for (size_t i = 0; i < length; i++)
            {
                rank_in_barrier_d[start + i] = i;
            }
        }
    }

    __device__ void step_2(int size)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (thID < size) // set_TID_d size
        {
            int TID = set_TID_d[thID];
            // bool blocked = true;
            // while (blocked)
            // {
            //     if (atomicCAS(&(state_d[TID].lock), 1, 0) == 1)
            //     {
            //         atomicMax(&(state_d[TID].rank), rank_in_barrier_d[thID]);
            //         atomicExch(&(state_d[TID].lock), 1);
            //         blocked = false;
            //     }
            // }
            atomicMax(&(state_d[TID].rank), rank_in_barrier_d[thID]);
        }
    }

    __device__ void step_3(int size, int rank)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        if (thID < size) // set_TID_d size
        {
            int TID = set_d[thID].TID;
            int loc = set_d[thID].Loc;
            int rank_current = state_d[TID].rank;
            if (rank_current == rank)
            {
                if (set_d[thID].op_type)
                {
                    current_2D_d[loc].data[0] += set_d[thID].local_set.data[0];
                    current_2D_d[loc].data[1] += set_d[thID].local_set.data[1];
                    current_2D_d[loc].data[2] += set_d[thID].local_set.data[2];
                    current_2D_d[loc].data[3] += set_d[thID].local_set.data[3];
                    current_2D_d[loc].data[4] += set_d[thID].local_set.data[4];
                    current_2D_d[loc].data[5] += set_d[thID].local_set.data[5];
                    current_2D_d[loc].data[6] += set_d[thID].local_set.data[6];
                    current_2D_d[loc].data[7] += set_d[thID].local_set.data[7];
                    current_2D_d[loc].data[8] += set_d[thID].local_set.data[8];
                    current_2D_d[loc].data[9] += set_d[thID].local_set.data[9];
                    current_2D_d[loc].data[10] += set_d[thID].local_set.data[10];
                    current_2D_d[loc].data[11] += set_d[thID].local_set.data[11];
                    current_2D_d[loc].data[12] += set_d[thID].local_set.data[12];
                    current_2D_d[loc].data[13] += set_d[thID].local_set.data[13];
                    current_2D_d[loc].data[14] += set_d[thID].local_set.data[14];
                    current_2D_d[loc].data[15] += set_d[thID].local_set.data[15];
                    current_2D_d[loc].data[16] += set_d[thID].local_set.data[16];
                    current_2D_d[loc].data[17] += set_d[thID].local_set.data[17];
                    current_2D_d[loc].data[18] += set_d[thID].local_set.data[18];
                    current_2D_d[loc].data[19] += set_d[thID].local_set.data[19];
                    current_2D_d[loc].data[20] += set_d[thID].local_set.data[20];
                }
                else
                {
                    current_2D_d[loc].data[0] = set_d[thID].local_set.data[0];
                    current_2D_d[loc].data[1] = set_d[thID].local_set.data[1];
                    current_2D_d[loc].data[2] = set_d[thID].local_set.data[2];
                    current_2D_d[loc].data[3] = set_d[thID].local_set.data[3];
                    current_2D_d[loc].data[4] = set_d[thID].local_set.data[4];
                    current_2D_d[loc].data[5] = set_d[thID].local_set.data[5];
                    current_2D_d[loc].data[6] = set_d[thID].local_set.data[6];
                    current_2D_d[loc].data[7] = set_d[thID].local_set.data[7];
                    current_2D_d[loc].data[8] = set_d[thID].local_set.data[8];
                    current_2D_d[loc].data[9] = set_d[thID].local_set.data[9];
                    current_2D_d[loc].data[10] = set_d[thID].local_set.data[10];
                    current_2D_d[loc].data[11] = set_d[thID].local_set.data[11];
                    current_2D_d[loc].data[12] = set_d[thID].local_set.data[12];
                    current_2D_d[loc].data[13] = set_d[thID].local_set.data[13];
                    current_2D_d[loc].data[14] = set_d[thID].local_set.data[14];
                    current_2D_d[loc].data[15] = set_d[thID].local_set.data[15];
                    current_2D_d[loc].data[16] = set_d[thID].local_set.data[16];
                    current_2D_d[loc].data[17] = set_d[thID].local_set.data[17];
                    current_2D_d[loc].data[18] = set_d[thID].local_set.data[18];
                    current_2D_d[loc].data[19] = set_d[thID].local_set.data[19];
                    current_2D_d[loc].data[20] = set_d[thID].local_set.data[20];
                }
            }
        }
    }
};

__global__ void kernel_1(Transaction T, int size)
{
    T.step_1(size);
}

__global__ void kernel_2(Transaction T, int size)
{
    T.step_2(size);
}

__global__ void kernel_3(Transaction T, int size, int rank)
{
    T.step_3(size, rank);
}

int main(int argc, char const *argv[])
{
    Transaction T;
    T.database.initial_data();
    T.copy_from_host_to_device_data();
    T.n_make.initial();
    T.p_make.initial();
    float time_all = 0.0;
    int commit = 0;
    for (size_t m = 0; m < EPOCH; m++)
    {
        time_t start = clock();
        T.random_new_query(m);

        // T.copy_from_host_to_device_tx();

        T.sort_v_and_id();

        kernel_1<<<(T.barrier_length_vec.size() / BLOCKSIZE + 1), BLOCKSIZE>>>(T, T.barrier_length_vec.size());
        cudaDeviceSynchronize();
        kernel_2<<<(T.set_TID_vec.size() / BLOCKSIZE + 1), BLOCKSIZE>>>(T, T.set_TID_vec.size());
        cudaDeviceSynchronize();
        T.copy_rank_device_to_host();
        for (size_t j = 0; j < T.rank_largest; j++)
        {
            kernel_3<<<(T.set_TID_vec.size() / BLOCKSIZE + 1), BLOCKSIZE>>>(T, T.set_TID_vec.size(), j);
            cudaDeviceSynchronize();
        }
        time_t end = clock();
        float time = (float)(end - start) / 1000000.0;
        if (m > WARMUP - 1 && m < EPOCH - WARMUP)
        {
            time_all += time;
            commit += BATCH_SIZE;
        }
        std::cout << "Epoch " << m << " cost : " << time << std::endl;
        if (m == EPOCH - 1)
        {
            std::cout << "TPS : " << commit / time_all << std::endl;
        }
        T.reset_gpu();
    }
    T.close_gpu();
    return 0;
}
