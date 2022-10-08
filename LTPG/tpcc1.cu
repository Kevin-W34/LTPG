#include "Query.h"
#include "Database.h"
#include "Execute1.h"
#include <vector>
#include <time.h>
#include "Predefine.h"

// flag size = TABLE_SIZE, hash tabel size = const HASH_SIZE, cooperative group size = const CG_SIZE

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
    int64_t ID_in_epoch = 0;
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
        // newquery.ID_in_epoch = ID_in_epoch;
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
        // newquery.ID_in_epoch = ID_in_epoch;
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
        for (size_t i = 0; i < HASH_SIZE; i++)
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
        cudaMemcpyAsync(current_2D_d, current_2D, sizeof(GLOBAL) * TABLE_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(flag_d, flag, sizeof(FLAG) * TABLE_SIZE, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
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
        for (size_t i = 0; i < HASH_SIZE; i++)
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
        cudaMemcpyAsync(flagcomp_d, flagcomp, sizeof(FLAGCOMP) * COMP_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(neworder_d, neworder_vec, sizeof(Set_n) * n_defined_cnt, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(payment_d, payment_vec, sizeof(Set_p) * p_defined_cnt, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
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
        cudaMemcpyAsync(neworder_new, neworder_d, sizeof(Set_n) * n_defined_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(payment_new, payment_d, sizeof(Set_p) * p_defined_cnt, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
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
            int state = payment_new[i].waw || (payment_new[i].war && payment_new[i].raw);
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

        // cudaFree(flag_d);
        // free(flag);

        cudaFree(config_n_d);
        cudaFree(config_p_d);
        free(config_n);
        free(config_p);
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

        int n_cnt = n_defined_cnt;
        int p_cnt = p_defined_cnt;

        int n_op = 8;
        int p_op = 7;

        int n_op_cnt_6 = 6 * n_cnt;
        // std::cout << "n_op_cnt_6 = " << n_op_cnt_6 << std::endl;
        int n_op_cnt_3 = 0;
        for (size_t i = 0; i < n_cnt; i++)
        {
            n_op_cnt_3 += 3 * neworder_vec[i].query.O_OL_CNT;
        }
        // std::cout << "n_op_cnt_3 = " << n_op_cnt_3 << std::endl;

        int n_op_cnt = n_op_cnt_6 + n_op_cnt_3;
        // std::cout << "n_op_cnt = " << n_op_cnt << std::endl;

        int p_op_cnt = 7 * p_cnt;
        // std::cout << "p_op_cnt = " << p_op_cnt << std::endl;

        int op_cnt = n_op_cnt + p_op_cnt;
        all_op_cnt = op_cnt;
        // std::cout << "op_cnt = " << op_cnt << std::endl;

        cudaDeviceProp cudade;
        cudaGetDeviceProperties(&cudade, 0);
        th_cnt = cudade.multiProcessorCount * cudade.maxThreadsPerMultiProcessor;
        // std::cout << "th_cnt = " << th_cnt << std::endl;

        int wp_cnt = th_cnt / 32;
        // std::cout << "wp_cnt = " << wp_cnt << std::endl;

        int n_wp_cnt = ((float)n_op_cnt / (float)op_cnt) * (float)wp_cnt;
        // std::cout << "n_wp_cnt = " << n_wp_cnt << std::endl;

        // int n_wp_per_op = (float)n_wp_cnt / (float)n_op;
        int n_wp_per_op_6;
        int n_wp_per_op_3;
        if (n_wp_cnt != 0)
        {
            n_wp_per_op_6 = (float)n_op_cnt_6 / (float)n_op_cnt * (float)n_wp_cnt / 5;
            n_wp_per_op_3 = (float)n_op_cnt_3 / (float)n_op_cnt * (float)n_wp_cnt / 3;
        }
        else
        {
            n_wp_per_op_6 = 0;
            n_wp_per_op_3 = 0;
        }
        // std::cout << "n_op_cnt_6 = " << n_op_cnt_6 / 5 << std::endl;
        // std::cout << "n_op_cnt_3 = " << n_op_cnt_3 / 3 << std::endl;
        // std::cout << "n_wp_per_op = " << n_wp_per_op << std::endl;

        int p_wp_cnt = ((float)p_op_cnt / (float)op_cnt) * (float)wp_cnt;
        // std::cout << "p_wp_cnt = " << p_wp_cnt << std::endl;

        int p_wp_per_op = (float)p_wp_cnt / (float)p_op;
        // std::cout << "p_wp_per_op = " << p_wp_per_op << std::endl;

        warp_split = n_wp_cnt;
        // std::cout << "warp_split = " << warp_split << std::endl;

        for (size_t i = 0; i < n_op * 4; i += 4)
        {
            config_n[i] = n_wp_per_op_6 * i / 4;
            config_n[i + 1] = n_wp_per_op_6;
            config_n[i + 2] = 0;
            config_n[i + 3] = n_cnt;
            if (i >= 20)
            {
                config_n[i] = config_n[i - 4] + config_n[i - 3];
                config_n[i + 1] = n_wp_per_op_3;
                config_n[i + 2] = 0;
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
        n_op_cnt_6 = n_op_cnt_6 / 5;
        n_op_cnt = n_op_cnt_6 + n_op_cnt_3;
        p_op_cnt = 4 * p_cnt;
        op_cnt = n_op_cnt + p_op_cnt;
        n_wp_cnt = ((float)n_op_cnt / (float)op_cnt) * (float)wp_cnt;
        if (n_wp_cnt != 0)
        {
            n_wp_per_op_6 = (float)n_op_cnt_6 / (float)n_op_cnt * (float)n_wp_cnt / 5;
            n_wp_per_op_3 = (float)n_op_cnt_3 / (float)n_op_cnt * (float)n_wp_cnt / 3;
        }
        else
        {
            n_wp_per_op_6 = 0;
            n_wp_per_op_3 = 0;
        }
        p_wp_cnt = ((float)p_op_cnt / (float)op_cnt) * (float)wp_cnt;
        p_wp_per_op = (float)p_wp_cnt / (float)4;
        for (size_t i = n_op * 4; i < (n_op + 4) * 4; i += 4) // write back config
        {
            config_n[i] = n_wp_per_op_6 * (i - n_op * 4) / 4;
            config_n[i + 1] = n_wp_per_op_6;
            config_n[i + 2] = 0;
            config_n[i + 3] = n_cnt;
            if (i >= 36)
            {
                config_n[i] = config_n[i - 4] + config_n[i - 3];
                config_n[i + 1] = n_wp_per_op_3;
                config_n[i + 2] = 0;
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
        cudaMemcpyAsync(config_n_d, config_n, sizeof(int) * CONFIG, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(config_p_d, config_p, sizeof(int) * CONFIG, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
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
    __device__ void resetFLAG(int th_cnt, int TABLE_SIZE)
    {
        int thID = threadIdx.x + blockDim.x * blockIdx.x;
        int i = thID;
        while (i < TABLE_SIZE)
        {
            flag_d[i].R_CNT = 0;
            flag_d[i].W_CNT = 0;
            for (size_t k = 0; k < HASH_SIZE; k++)
            {
                flag_d[i].TID_LIST_R[k] = 1 << 31 - 1;
                flag_d[i].TID_LIST_W[k] = 1 << 31 - 1;
                flag_d[i].lock_R[k] = 1;
                flag_d[i].lock_W[k] = 1;
            }
            if (i < WAREHOUSE_SIZE + DISTRICT_SIZE)
            {
                flagcomp_d[i].R_CNT = 0;
                flagcomp_d[i].W_CNT = 0;
                for (size_t k = 0; k < HASH_SIZE; k++)
                {
                    flagcomp_d[i].TID_LIST_R[k] = 1 << 31 - 1;
                    flagcomp_d[i].TID_LIST_W[k] = 1 << 31 - 1;
                    flagcomp_d[i].lock_R[k] = 1;
                    flagcomp_d[i].lock_W[k] = 1;
                }
            }
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
        PAYMENT_QUERY::execute_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
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

// __global__ void kernel_3(Transaction T)
// {
//     T.resetFLAG(T.th_cnt, T.TABLE_SIZE);
// }
// __global__ void kernel_4(Transaction T)
// {
//     int thID = threadIdx.x + blockDim.x * blockIdx.x;
//     int wID = thID / 32;
//     if (wID < T.warp_split) // T.warp_split
//     {
//         NEWORDER_QUERY::execute_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
//     }
//     else
//     {
//         PAYMENT_QUERY::execute_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
//     }
//     __syncthreads();
//     __threadfence();
//     if (wID < T.warp_split)
//     {
//         // printf("check n %d:%d\n", wID, thID);
//         NEWORDER_QUERY::check_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
//     }
//     else
//     {
//         // printf("check p %d:%d\n", wID, thID);
//         PAYMENT_QUERY::check_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
//     }
//     __syncthreads();
//     __threadfence();
//     if (wID < T.warp_split)
//     {
//         NEWORDER_QUERY::write_n(T.config_n_d, T.current_2D_d, T.neworder_d, T.flag_d, T.flagcomp_d);
//     }
//     else
//     {
//         PAYMENT_QUERY::write_p(T.config_p_d, T.current_2D_d, T.payment_d, T.flag_d, T.flagcomp_d);
//     }
//     __syncthreads();
//     __threadfence();
//     T.resetFLAG(T.th_cnt, TABLE_SIZE_1D);
// }

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

        kernel_0<<<68 * 16, 64>>>(tx); // 68*32*32
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_0 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_0_t = clock();
        kernel_1<<<68 * 16, 64>>>(tx); // 68*32*32
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_1 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_1_t = clock();
        kernel_2<<<68 * 16, 64>>>(tx); // 68*32*32
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "kernel_2 CUDA Error : " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();
        time_t kernel_2_t = clock();

        // kernel_3<<<68 * 16, 64>>>(tx);
        // err = cudaGetLastError();
        // if (err != cudaSuccess)
        // {
        //     std::cout << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
        // }
        // cudaDeviceSynchronize();
        // time_t kernel_3_t = clock();
        // kernel_4<<<68 * 16, 64>>>(tx);
        // cudaDeviceSynchronize();
        // time_t kernel_4_t = clock();

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