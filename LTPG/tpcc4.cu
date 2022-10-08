#include "Query.h"
#include "Database.h"
#include "Execute4.h"
#include <vector>
#include <time.h>
#include <math.h>
#include <thrust/sort.h>
#include "Predefine.h"

// flag size = TABLE_SIZE, hash tabel size = dynamic HASH_SIZE, cooperative group size = dynamic CG_SIZE

class Transaction
{
public:
    // std::vector<Set_n>
    Set_n *neworder_vec;
    Set_n *neworder_d;
    Set_n *neworder_new;
    // std::vector<Set_p>
    Set_p *payment_vec;
    Set_p *payment_d;
    Set_p *payment_new;

    int *w_ID_d;
    int *d_ID_d;
    int *c_ID_d;

    int *w_Loc_d;
    int *d_Loc_d;
    int *c_Loc_d;

    GLOBAL *current_2D;
    GLOBAL *current_2D_d;
    int *config_n;
    int *config_n_d;
    int *config_p;
    int *config_p_d;
    FLAG *flag;
    FLAG *flag_d;
    FLAGCOMP *flagcomp;
    FLAGCOMP *flagcomp_d;
    Random random;
    Database database;
    MakeNewOrder n_make;
    MakePayment p_make;

    int64_t TID_global = 0;

    int TABLE_SIZE = TABLE_SIZE_1D;
    int COMP_SIZE = WAREHOUSE_SIZE + DISTRICT_SIZE;

    int all_op_cnt = 0;
    int th_cnt;
    int n_commit;
    int p_commit;
    int all_commit = 0;
    int n_commit_all = 0;
    int p_commit_all = 0;
    int n_remain_cnt = 0;
    int p_remain_cnt = 0;
    int warp_split = 0;
    int n_defined_cnt = BATCH_SIZE * NEWORDERPERCENT / 100;
    int p_defined_cnt = BATCH_SIZE - n_defined_cnt;

    void random_new_query(int NO)
    {
        std::cout << "\nstart make new\n";
        if (NO == 0)
        {
            neworder_vec = new Set_n[n_defined_cnt];
            payment_vec = new Set_p[p_defined_cnt];
        }
        int nsize = n_remain_cnt;
        int psize = p_remain_cnt;
        // n_remain_cnt = 0;
        // p_remain_cnt = 0;
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
        TID_global += 1;
        // ID_in_epoch += 1;
        neworder_vec[Loc] = newquery;
    }
    void new_payment(int Loc)
    {

        Payment query = p_make.make();
        Set_p newquery;
        newquery.query = query;
        newquery.TID = TID_global;
        newquery.ID = Loc;
        TID_global += 1;
        // ID_in_epoch += 1;
        payment_vec[Loc] = newquery;
    }

    void copy_from_host_to_device_data()
    {

        std::cout << "\nstart copy data from host to device\n";
        time_t start_cpy_data = clock();
        current_2D = database.getSnapShot_2D();

        flag = new FLAG[TABLE_SIZE];
        FLAG temp;
        for (size_t i = 0; i < NORMAL_HASH_SIZE; i++)
        {
            temp.lock_R[i] = 1;
            temp.lock_W[i] = 1;
            temp.TID_LIST_W[i] = 1 << 31 - 1;
            temp.TID_LIST_R[i] = 1 << 31 - 1;
        }
        for (size_t i = 0; i < TABLE_SIZE; i++)
        {
            flag[i] = temp;
        }

        cudaMalloc((void **)&current_2D_d, sizeof(GLOBAL) * TABLE_SIZE);
        cudaMalloc((void **)&flag_d, sizeof(FLAG) * TABLE_SIZE);
        cudaMemcpy(current_2D_d, current_2D, sizeof(GLOBAL) * TABLE_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(flag_d, flag, sizeof(FLAG) * TABLE_SIZE, cudaMemcpyHostToDevice);
        time_t end_cpy_data = clock();
        std::cout << "copy data cost = " << (float)(end_cpy_data - start_cpy_data) / 1000000.0 << std::endl;
        std::cout << "end copy data from host to device\n\n";
        // std::cout << current_2D[TABLE_SIZE_1D - 1].data[0] << std::endl;
        // std::cout << current_2D[TABLE_SIZE_1D - 1].data[1] << std::endl;
        // std::cout << current_2D[TABLE_SIZE_1D - 1].data[2] << std::endl;
        // std::cout << current_2D[TABLE_SIZE_1D - 1].data[3] << std::endl;
        // std::cout << current_2D[TABLE_SIZE_1D - 1].data[4] << std::endl;
    }

    void copy_from_host_to_device_tx()
    {
        std::cout << "\nstart copy tx from host to device\n";
        flagcomp = new FLAGCOMP[COMP_SIZE];
        FLAGCOMP tempcomp;
        for (size_t i = 0; i < COMPETITION_HASH_SIZE; i++)
        {
            tempcomp.lock_R[i] = 1;
            tempcomp.lock_W[i] = 1;
            tempcomp.TID_LIST_R[i] = 1 << 31 - 1;
            tempcomp.TID_LIST_W[i] = 1 << 31 - 1;
        }
        for (size_t i = 0; i < COMP_SIZE; i++)
        {
            flagcomp[i] = tempcomp;
        }
        cudaMalloc((void **)&flagcomp_d, sizeof(FLAGCOMP) * COMP_SIZE);
        cudaMalloc((void **)&neworder_d, sizeof(Set_n) * n_defined_cnt);
        cudaMalloc((void **)&payment_d, sizeof(Set_p) * p_defined_cnt);
        cudaMalloc((void **)&w_ID_d, sizeof(int) * p_defined_cnt);
        cudaMalloc((void **)&d_ID_d, sizeof(int) * p_defined_cnt);
        cudaMalloc((void **)&c_ID_d, sizeof(int) * p_defined_cnt);
        cudaMalloc((void **)&w_Loc_d, sizeof(int) * p_defined_cnt);
        cudaMalloc((void **)&d_Loc_d, sizeof(int) * p_defined_cnt);
        cudaMalloc((void **)&c_Loc_d, sizeof(int) * p_defined_cnt);
        cudaMemcpy(flagcomp_d, flagcomp, sizeof(FLAGCOMP) * COMP_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(neworder_d, neworder_vec, sizeof(Set_n) * n_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(payment_d, payment_vec, sizeof(Set_p) * p_defined_cnt, cudaMemcpyHostToDevice);
        std::cout << "end copy tx from host to device\n\n";
    }

    void copy_from_device_to_host_data()
    {
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
        // free(neworder_vec);
        // free(payment_vec);
        // neworder_vec = new Set_n[n_defined_cnt];
        // payment_vec = new Set_p[p_defined_cnt];
        n_commit = 0;
        int n_abort = 0;
        p_commit = 0;
        int p_abort = 0;
        n_remain_cnt = 0;
        p_remain_cnt = 0;
        for (size_t i = 0; i < pre_neworder_size; i++)
        {
            // int state = neworder_new[i].state;
            int state = neworder_new[i].waw || (neworder_new[i].war && neworder_new[i].raw);
            // std::cout << "No." << neworder_new[i].TID << " waw = " << neworder_new[i].waw << std::endl;
            if (state > 0)
            {
                // neworder_new[i].raw = 0;
                // neworder_new[i].waw = 0;
                // neworder_new[i].war = 0;
                // neworder_vec[n_remain_cnt] = neworder_new[i];
                // n_remain_cnt += 1;
                n_abort += 1;
                // std::cout << "No." << neworder_new[i].TID << " waw = " << neworder_new[i].waw << std::endl;
            }
            else
            {
                n_commit += 1;
            }
        }

        // std::cout << "No." << payment_new[19].TID << " waw = " << payment_new[19].waw << std::endl;
        for (size_t i = 0; i < pre_payment_size; i++)
        {
            int state = payment_new[i].state;
            // int state = payment_new[i].waw || (payment_new[i].war && payment_new[i].raw);
            // std::cout << "No." << payment_new[i].TID << " waw = " << payment_new[i].waw << std::endl;
            if (state > 0)
            {
                // payment_new[i].war = 0;
                // payment_new[i].raw = 0;
                // payment_new[i].waw = 0;
                // payment_vec[p_remain_cnt] = payment_new[i];
                // p_remain_cnt += 1;
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
        cudaFree(flagcomp_d);
        free(neworder_new);
        free(payment_new);
        free(flagcomp);

        cudaFree(config_n_d);
        cudaFree(config_p_d);
        free(config_n);
        free(config_p);

        cudaFree(w_ID_d);
        cudaFree(d_ID_d);
        cudaFree(c_ID_d);
        cudaFree(w_Loc_d);
        cudaFree(d_Loc_d);
        cudaFree(c_Loc_d);
        std::cout << "abort neworder = " << n_abort << std::endl;
        std::cout << "abort payment = " << p_abort << std::endl;
        std::cout << "commit neworder = " << n_commit << std::endl;
        std::cout << "commit payment = " << p_commit << std::endl;
        std::cout << "end copy tx from device to host\n\n";
    }

    void config_gpu()
    {
        std::cout << "\nstart config\n";
        config_n = new int[CONFIG]; // [warp start-warp length-tx start-tx length]
        config_p = new int[CONFIG];
        all_op_cnt = 0;
        for (size_t i = 0; i < CONFIG; i++)
        {
            config_n[i] = 0;
            config_p[i] = 0;
        }
        cudaDeviceProp cudade;
        cudaGetDeviceProperties(&cudade, 0);
        th_cnt = cudade.multiProcessorCount * cudade.maxThreadsPerMultiProcessor;
        // std::cout << "th_cnt = " << th_cnt << std::endl;

        int wp_cnt = th_cnt / 32;
        int n_cnt = n_defined_cnt;
        int p_cnt = p_defined_cnt;

        int n_op = 8;
        int p_op = 7;

        int n_op_cnt_5 = 5 * n_cnt;
        // std::cout << "n_op_cnt_5 = " << n_op_cnt_5 << std::endl;
        int n_op_cnt_3 = 0;
        for (size_t i = 0; i < n_cnt; i++)
        {
            n_op_cnt_3 += 3 * neworder_vec[i].query.O_OL_CNT;
        }
        // std::cout << "n_op_cnt_3 = " << n_op_cnt_3 << std::endl;

        int n_op_cnt = n_op_cnt_5 + n_op_cnt_3;
        // std::cout << "n_op_cnt = " << n_op_cnt << std::endl;

        int p_op_cnt = 7 * p_cnt;
        // std::cout << "p_op_cnt = " << p_op_cnt << std::endl;

        int op_cnt = n_op_cnt + p_op_cnt;
        all_op_cnt = op_cnt;
        // std::cout << "op_cnt = " << op_cnt << std::endl;

        // std::cout << "wp_cnt = " << wp_cnt << std::endl;

        int n_wp_cnt = ((float)n_op_cnt / (float)op_cnt) * (float)wp_cnt;
        // std::cout << "n_wp_cnt = " << n_wp_cnt << std::endl;

        // int n_wp_per_op = (float)n_wp_cnt / (float)n_op;
        int n_wp_per_op_5;
        int n_wp_per_op_3;
        if (n_wp_cnt != 0)
        {
            n_wp_per_op_5 = (float)n_op_cnt_5 / (float)n_op_cnt * (float)n_wp_cnt / 5;
            n_wp_per_op_3 = (float)n_op_cnt_3 / (float)n_op_cnt * (float)n_wp_cnt / 3;
        }
        else
        {
            n_wp_per_op_5 = 1;
            n_wp_per_op_3 = 1;
        }
        // std::cout << "n_op_cnt_5 = " << n_op_cnt_5 / 5 << std::endl;
        // std::cout << "n_op_cnt_3 = " << n_op_cnt_3 / 3 << std::endl;
        // std::cout << "n_wp_per_op = " << n_wp_per_op << std::endl;

        int p_wp_cnt = ((float)p_op_cnt / (float)op_cnt) * (float)wp_cnt;
        // std::cout << "p_wp_cnt = " << p_wp_cnt << std::endl;

        int p_wp_per_op = (float)p_wp_cnt / (float)p_op;
        // std::cout << "p_wp_per_op = " << p_wp_per_op << std::endl;

        warp_split = n_wp_cnt;
        // std::cout << "warp_split = " << warp_split << std::endl;

        int cgsize = n_cnt / n_wp_per_op_3;
        std::cout << "cgsize = " << cgsize << std::endl;
        cgsize = cgsize > 16 ? 16 : cgsize;
        cgsize = log2(cgsize) > 0 ? log2(cgsize) : 1;
        // cgsize = pow(2, cgsize);
        cgsize = 32 / pow(2, cgsize);
        std::cout << "cgsize = " << cgsize << std::endl;
        for (size_t i = 0; i < n_op * 4; i += 4)
        {
            config_n[i] = n_wp_per_op_5 * i / 4;
            config_n[i + 1] = n_wp_per_op_5;
            config_n[i + 2] = 0;
            config_n[i + 3] = n_cnt;
            if (i >= 20)
            {
                config_n[i] = config_n[i - 4] + config_n[i - 3];
                config_n[i + 1] = n_wp_per_op_3;
                config_n[i + 2] = cgsize;
                config_n[i + 3] = n_cnt;
            }
            // std::cout << config_n[i] << "-" << config_n[i + 1] << "-" << config_n[i + 2] << "-" << config_n[i + 3] << std::endl;
        }
        // std::cout << "======================================\n";
        for (size_t i = 0; i < p_op * 4; i += 4)
        {
            config_p[i] = warp_split + p_wp_per_op * i / 4;
            config_p[i + 1] = p_wp_per_op;
            config_p[i + 2] = 0;
            config_p[i + 3] = p_cnt;
            // std::cout << config_p[i] << "-" << config_p[i + 1] << "-" << config_p[i + 2] << "-" << config_p[i + 3] << std::endl;
        }
        // std::cout << "======================================\n";
        n_op_cnt_5 = n_op_cnt_5 / 5;
        n_op_cnt = n_op_cnt_5 + n_op_cnt_3;
        p_op_cnt = 4 * p_cnt;
        op_cnt = n_op_cnt + p_op_cnt;
        n_wp_cnt = ((float)n_op_cnt / (float)op_cnt) * (float)wp_cnt;
        if (n_wp_cnt != 0)
        {
            n_wp_per_op_5 = (float)n_op_cnt_5 / (float)n_op_cnt * (float)n_wp_cnt;
            n_wp_per_op_3 = (float)n_op_cnt_3 / (float)n_op_cnt * (float)n_wp_cnt / 3;
        }
        else
        {
            n_wp_per_op_5 = 1;
            n_wp_per_op_3 = 1;
        }
        p_wp_cnt = ((float)p_op_cnt / (float)op_cnt) * (float)wp_cnt;
        p_wp_per_op = (float)p_wp_cnt / (float)4;

        cgsize = n_cnt / n_wp_per_op_3;
        cgsize = cgsize > 16 ? 16 : cgsize;
        cgsize = log2(cgsize) > 0 ? log2(cgsize) : 1;
        // cgsize = pow(2, cgsize);
        cgsize = 32 / pow(2, cgsize);
        // std::cout << "cgsize = " << cgsize << std::endl;
        for (size_t i = n_op * 4; i < (n_op + 4) * 4; i += 4) // write back config
        {
            config_n[i] = n_wp_per_op_5 * (i - n_op * 4) / 4;
            config_n[i + 1] = n_wp_per_op_5;
            config_n[i + 2] = 0;
            config_n[i + 3] = n_cnt;
            if (i >= 32)
            {
                config_n[i] = config_n[i - 4] + config_n[i - 3];
                config_n[i + 1] = n_wp_per_op_3;
                config_n[i + 2] = cgsize;
                config_n[i + 3] = n_cnt;
            }

            // std::cout << config_n[i] << "-" << config_n[i + 1] << "-" << config_n[i + 2] << "-" << config_n[i + 3] << std::endl;
        }
        // std::cout << "======================================\n";
        for (size_t i = p_op * 4; i < (p_op + 4) * 4; i += 4)
        { // write back config
            config_p[i] = warp_split + p_wp_per_op * (i - p_op * 4) / 4;
            config_p[i + 1] = p_wp_per_op;
            config_p[i + 2] = 0;
            config_p[i + 3] = p_cnt;
            // std::cout << config_p[i] << "-" << config_p[i + 1] << "-" << config_p[i + 2] << "-" << config_p[i + 3] << std::endl;
        }
        // std::cout << "======================================\n";

        cudaMalloc((void **)&config_n_d, sizeof(int) * CONFIG);
        cudaMalloc((void **)&config_p_d, sizeof(int) * CONFIG);
        cudaMemcpy(config_n_d, config_n, sizeof(int) * CONFIG, cudaMemcpyHostToDevice);
        cudaMemcpy(config_p_d, config_p, sizeof(int) * CONFIG, cudaMemcpyHostToDevice);
        std::cout << "end config gpu\n\n";
    }

    void clean()
    {
        std::cout << "\nstart clean\n";
        std::cout << "end clean\n\n";
    }
    void close_gpu()
    {
        free(neworder_vec);
        free(payment_vec);

        cudaFree(current_2D_d);
        cudaFree(flag_d);
        free(current_2D);
        free(flag);
    }

    void sort()
    {
        int *w_ID;
        int *d_ID;
        int *c_ID;
        int *w_Loc;
        int *d_Loc;
        int *c_Loc;
        w_ID = new int[p_defined_cnt];
        d_ID = new int[p_defined_cnt];
        c_ID = new int[p_defined_cnt];
        w_Loc = new int[p_defined_cnt];
        d_Loc = new int[p_defined_cnt];
        c_Loc = new int[p_defined_cnt];
        cudaMemcpy(w_ID, w_ID_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_ID, d_ID_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(c_ID, c_ID_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(w_Loc, w_Loc_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_Loc, d_Loc_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(c_Loc, c_Loc_d, sizeof(int) * p_defined_cnt, cudaMemcpyDeviceToHost);
        thrust::sort_by_key(w_Loc, w_Loc + p_defined_cnt, w_ID);
        thrust::sort_by_key(d_Loc, d_Loc + p_defined_cnt, d_ID);
        thrust::sort_by_key(c_Loc, c_Loc + p_defined_cnt, c_ID);
        int barrier = 0;
        for (size_t i = 0; i < p_defined_cnt; i++)
        {
            if (w_Loc[i] != w_Loc[barrier])
            {
                thrust::sort_by_key(w_ID + barrier, w_ID + i - 1, w_Loc + barrier);
                barrier = i;
            }
        }
        barrier = 0;
        for (size_t i = 0; i < p_defined_cnt; i++)
        {
            if (d_Loc[i] != d_Loc[barrier])
            {
                thrust::sort_by_key(d_ID + barrier, d_ID + i - 1, d_Loc + barrier);
                barrier = i;
            }
        }
        barrier = 0;
        for (size_t i = 0; i < p_defined_cnt; i++)
        {
            if (c_Loc[i] != c_Loc[barrier])
            {
                thrust::sort_by_key(c_ID + barrier, c_ID + i - 1, c_Loc + barrier);
                barrier = i;
            }
        }
        // for (size_t i = 0; i < p_defined_cnt; i++)
        // {
        //     std::cout << w_ID[i] << " " << w_Loc[i] << std::endl;
        // }

        cudaMemcpy(w_ID_d, w_ID, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ID_d, d_ID, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(c_ID_d, c_ID, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(w_Loc_d, w_Loc, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Loc_d, d_Loc, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpy(c_Loc_d, c_Loc, sizeof(int) * p_defined_cnt, cudaMemcpyHostToDevice);
        free(w_ID);
        free(d_ID);
        free(c_ID);
        free(w_Loc);
        free(d_Loc);
        free(c_Loc);
    }

    __device__ void resetFLAG(int th_cnt, int TABLE_SIZE)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int i = thID;
        while (i < TABLE_SIZE)
        {
            flag_d[i].R_CNT = 0;
            flag_d[i].W_CNT = 0;

            flag_d[i].TID_LIST_R[0] = 1 << 31 - 1;
            flag_d[i].TID_LIST_W[0] = 1 << 31 - 1;
            flag_d[i].lock_R[0] = 1;
            flag_d[i].lock_W[0] = 1;

            // if (i < WAREHOUSE_SIZE + DISTRICT_SIZE)
            // {
            //     flagcomp_d[i].R_CNT = 0;
            //     flagcomp_d[i].W_CNT = 0;
            //     for (size_t k = 0; k < COMPETITION_HASH_SIZE; k++)
            //     {
            //         flagcomp_d[i].TID_LIST_R[k] = 1 << 31 - 1;
            //         flagcomp_d[i].TID_LIST_W[k] = 1 << 31 - 1;
            //         flagcomp_d[i].lock_R[k] = 1;
            //         flagcomp_d[i].lock_W[k] = 1;
            //     }
            // }
            i += th_cnt;
        }
    }
};

__global__ void kernel_0(Transaction T)
{
    int thID = threadIdx.x + blockDim.x * blockIdx.x;
    int wID = thID / 32;
    // if (thID == 0)
    // {
    //     printf("\nstart kernel_0\n\n");
    // }

    if (wID < T.warp_split) // T.warp_split
    {
        // printf("exe n %d:%d\n", wID, thID);
        NEWORDER_QUERY::execute_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
    }
    else
    {
        // printf("exe p %d:%d\n", wID, thID);
        PAYMENT_QUERY::execute_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d, T.w_ID_d, T.d_ID_d, T.c_ID_d, T.w_Loc_d, T.d_Loc_d, T.c_Loc_d);
    }
}
__global__ void kernel_1(Transaction T)
{
    int thID = threadIdx.x + blockDim.x * blockIdx.x;
    int wID = thID / 32;
    // if (thID == 0)
    // {
    //     printf("\nstart kernel_1\n\n");
    // }
    if (wID < T.warp_split)
    {
        // printf("check n %d:%d\n", wID, thID);
        NEWORDER_QUERY::check_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
    }
    else
    {
        // printf("check p %d:%d\n", wID, thID);
        PAYMENT_QUERY::check_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
    }
}

__global__ void kernel_2(Transaction T)
{
    int thID = threadIdx.x + blockDim.x * blockIdx.x;
    int wID = thID / 32;
    // if (thID == 0)
    // {
    //     printf("\nstart kernel_2\n\n");
    // }
    if (wID < T.warp_split)
    {
        NEWORDER_QUERY::write_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
    }
    else
    {
        PAYMENT_QUERY::write_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
    }
    // __threadfence();
    T.resetFLAG(T.th_cnt, T.TABLE_SIZE);
}

__global__ void kernel_3(Transaction T)
{
    // int thID = threadIdx.x + blockDim.x * blockIdx.x;
    // if (thID == 0)
    // {
    //     printf("\nstart kernel_2\n\n");
    // }
    if (NEWORDERPERCENT < 100)
    {
        PAYMENT_QUERY::exec_hco(T.config_p_d, T.current_2D_d, T.payment_d, T.p_defined_cnt, T.w_ID_d, T.d_ID_d, T.c_ID_d, T.w_Loc_d, T.d_Loc_d, T.c_Loc_d);
    }
    // __threadfence();
    // T.resetFLAG(T.th_cnt, T.TABLE_SIZE);
}

int main(int argc, char const *argv[])
{
    Transaction tx;
    time_t start = clock();
    tx.database.initial_data();
    tx.copy_from_host_to_device_data();
    time_t end_copy_data_HTD = clock();
    std::cout << "initial and copy data cost : " << (float)(end_copy_data_HTD - start) / 1000000.0 << std::endl;
    std::cout << "\nstart initial\n";
    tx.p_make.initial();
    tx.n_make.initial();
    std::cout << "end initial\n\n";
    float time_all = 0.0;
    float time_k0 = 0.0;
    float time_k1 = 0.0;
    float time_k2 = 0.0;
    for (int m = 0; m < EPOCH; m++)
    {
        time_t end1 = clock();
        tx.random_new_query(m);
        time_t end2 = clock();
        std::cout << "random new query cost : " << (float)(end2 - end1) / 1000000.0 << std::endl;
        tx.copy_from_host_to_device_tx();
        time_t end3 = clock();
        std::cout << "copy tx cost : " << (float)(end3 - end2) / 1000000.0 << std::endl;
        tx.config_gpu();
        time_t end4 = clock();
        std::cout << "config gpu cost : " << (float)(end4 - end3) / 1000000.0 << std::endl;
        std::cout << "======================================\n";

        kernel_0<<<68 * 1024 / BLOCKSIZE, BLOCKSIZE>>>(tx); // 68*32*32
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_0 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_0_t = clock();
        tx.sort();
        kernel_1<<<68 * 1024 / BLOCKSIZE, BLOCKSIZE>>>(tx); // 68*32*32
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_1 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_1_t = clock();
        kernel_2<<<68 * 1024 / BLOCKSIZE, BLOCKSIZE>>>(tx); // 68*32*32
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_2 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_2_t = clock();

        kernel_3<<<68 * 1024 / BLOCKSIZE, BLOCKSIZE>>>(tx);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_3_t = clock();

        std::cout << "======================================\n";
        tx.copy_from_device_to_host_tx(m);
        time_t end10 = clock();
        std::cout << "kernel_0 cost : " << (float)(kernel_0_t - end4) / 1000000.0 << std::endl;
        std::cout << "kernel_1 cost : " << (float)(kernel_1_t - kernel_0_t) / 1000000.0 << std::endl;
        std::cout << "kernel_2 cost : " << (float)(kernel_2_t - kernel_1_t) / 1000000.0 << std::endl;
        std::cout << "all kernels cost : " << (float)(kernel_2_t - end4) / 1000000.0 << std::endl;
        float time = (float)(end10 - end1) / 1000000.0;
        std::cout << "copy tx device to host : " << (float)(end10 - kernel_2_t) / 1000000.0 << std::endl;
        std::cout << "NO." << m << " Epoch cost : " << time << std::endl;
        if (m > WARMUP - 1 && m < EPOCH - WARMUP)
        {
            time_all += time;
            time_k0 += (float)(kernel_0_t - end4) / 1000000.0;
            time_k1 += (float)(kernel_1_t - kernel_0_t) / 1000000.0;
            time_k2 += (float)(kernel_2_t - kernel_1_t) / 1000000.0;
        }
        if (m == EPOCH - 1)
        {
            std::cout << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs cost = " << time_all << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs kernel 0 cost = " << time_k0 << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs kernel 1 cost = " << time_k1 << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs kernel 2 cost = " << time_k2 << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs commit all = " << tx.all_commit << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs commit neworder = " << tx.n_commit_all << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs commit payment = " << tx.p_commit_all << std::endl;
            std::cout << EPOCH - WARMUP * 2 << "epochs TPS = " << (float)tx.all_commit / time_all << std::endl;
        }
        std::cout << "======================================\n";
        std::cout << "======================================\n";
    }
    tx.close_gpu();
    return 0;
}