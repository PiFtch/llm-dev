#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
// #include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
// #include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

#define NEG_INF -1e9

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

void flash_attn_sycl(float* Q, float* K, float* V,
                     float* O, float* m, float* l,
                     int batch_size, int seq_len,
                     int d_model, int Br, int Bc,
                     float* mask){
    queue q;

    q.submit([&](handler& h) {
        // dispath in 3-dimensional grid
        // XY the single batch
        h.parallel_for(range<3>(seq_len / Br, seq_len / Bc, batch_size), [=](id<3> idx) {
            int batch_idx = idx[2];
            int row_start = idx[0] * Br;
            int block_start_Bc = idx[1] * Bc;

            // Now we are processing 1 block
            if (row_start < seq_len && block_start_Bc < seq_len) {
                for (int i = 0; i < Br; ++i) {
                    for (int j = 0; j < Bc; ++j) {
                        int q_idx = batch_idx * seq_len * d_model + (row_start + i) * d_model;
                        int k_idx = batch_idx * seq_len * d_model + (block_start_Bc + j) * d_model;
                        int v_idx = k_idx;

                        float Sij = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            Sij += Q[q_idx + d] * K[k_idx + d];
                        }

                        if (mask != nullptr) {
                            int mask_idx = batch_idx * seq_len * seq_len + (row_start + i) * seq_len + (block_start_Bc + j);
                            if (mask[mask_idx] == 0) {
                                Sij = NEG_INF;
                            }
                        }

                        int m_idx = batch_idx * seq_len + row_start + i;
                        atomic_ref<float, memory_order::relaxed, memory_scope::device> m_atomic(m[m_idx]);
                        // atomic_fetch_max(&m[m_idx], Sij);
                        m_atomic.fetch_max(Sij);

                        int l_idx = batch_idx * seq_len + row_start + i;
                        atomic_ref<float, memory_order::relaxed, memory_scope::device> l_atomic(l[l_idx]);
                        l_atomic.fetch_add(sycl::exp(Sij - m[m_idx]));

                        int o_idx = batch_idx * seq_len * d_model + (row_start + i) * d_model;
                        for (int d = 0; d < d_model; ++d) {
                            atomic_ref<float, memory_order::relaxed, memory_scope::device> O_atomic(O[o_idx + d]);
                            O_atomic.fetch_add(sycl::exp(Sij - m[m_idx]) * V[v_idx + d]);
                        }
                    }
                }
            }
        });
    }).wait();
}

int main() {
    // 示例参数
    int batch_size = 1;
    // 暂不考虑batch
    int seq_len = 4;
    int d_model = 10;
    int Br = 2;
    int Bc = 1;

    int Tr = seq_len / Br;
    int Tc = seq_len / Bc;

    // const int SG_SIZE = 10;

    range<2> global_range(Tr, Tc);
    range<2> local_range(1, 1);
    // range<2> global_range(10, 10);
    // range<2> local_range(2, SG_SIZE);

    // 分配内存
    std::vector<float> Q(batch_size * seq_len * d_model, 1.0f);
    std::vector<float> K(batch_size * seq_len * d_model, 1.0f);
    std::vector<float> V(batch_size * seq_len * d_model, 1.0f);
    std::vector<float> O(batch_size * seq_len * d_model, 0.0f);
    std::vector<float> m(batch_size * seq_len, NEG_INF);
    std::vector<float> l(batch_size * seq_len, 0.0f);
    std::vector<float> mask(batch_size * seq_len * seq_len, 1.0f);

    std::vector<float> temp(10 * 10, -1.0f);

    // 调用 SYCL 内核
    // flash_attn_sycl(Q.data(), K.data(), V.data(), O.data(), m.data(), l.data(), batch_size, seq_len, d_model, Br, Bc, mask.data());
    auto devices = device::get_devices();
    std::cout << "Available SYCL devices:" << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  " << dev.get_info<info::device::name>() << std::endl;
    }

    queue q;
    try {
        buffer<float, 2> Q_buf(Q.data(), range<2>(seq_len, d_model));
        buffer<float, 2> K_buf(K.data(), range<2>(seq_len, d_model));
        buffer<float, 2> V_buf(V.data(), range<2>(seq_len, d_model));

        buffer<float, 2> temp_buf(temp.data(), range<2>(10, 10));

        // 声明子缓冲区, sub_offset 为子缓冲区的起始位置, sub_range 为子缓冲区的大小
        std::vector<buffer<float, 2>> sub_buf_Q_list;
        std::vector<buffer<float, 2>> sub_buf_K_list;
        std::vector<buffer<float, 2>> sub_buf_V_list;
        
        for (int i = 0; i < Tr; i++) {
            id<2> sub_offset(i * Br, 0);
            range<2> sub_range(Br, d_model);
            sub_buf_Q_list.emplace_back(Q_buf, sub_offset, sub_range);
        }
        for (int i = 0; i < Tc; i++) {
            id<2> sub_offset(i * Bc, 0);
            range<2> sub_range(Bc, d_model);
            sub_buf_K_list.emplace_back(K_buf, sub_offset, sub_range);
            sub_buf_V_list.emplace_back(V_buf, sub_offset, sub_range);
        }
        
        q.submit([&](handler& h) {
            // 创建访问器
            std::vector<accessor<float, 2, access::mode::read_write, access::target::device>> subQ_acc_list;
            std::vector<accessor<float, 2, access::mode::read_write, access::target::device>> subK_acc_list;
            std::vector<accessor<float, 2, access::mode::read_write, access::target::device>> subV_acc_list;

            for (auto& sub_buf_Q : sub_buf_Q_list) {
                subQ_acc_list.emplace_back(sub_buf_Q.get_access<access::mode::read_write>(h));
            }
            for (auto& sub_buf_K : sub_buf_K_list) {
                subK_acc_list.emplace_back(sub_buf_K.get_access<access::mode::read_write>(h));
            }
            for (auto& sub_buf_V : sub_buf_V_list) {
                subV_acc_list.emplace_back(sub_buf_V.get_access<access::mode::read_write>(h));
            }

            auto temp_acc = temp_buf.get_access<access::mode::read_write>(h);

            h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) /*[[sycl::reqd_sub_group_size(1)]]*/ {
                sub_group sg = item.get_sub_group();
                int sg_id = sg.get_group_id();

                joint_matrix<sub_group, half, use::a, 2, 2, layout::row_major> tA;
                atomic_ref<float, memory_order::relaxed, memory_scope::device> temp_atomic(temp_acc[sg_id][sg_id]);
                temp_atomic.fetch_add(3.0f);

                // global_idx is block
                // get the Qij, Kij and Vij
                int wg_idx = item.get_global_id(0);
                int wg_idy = item.get_global_id(1);
                int local_idx = item.get_local_id(0);
                int local_idy = item.get_local_id(1);

                /* 对于每个work-item来说，local_id只可能是local_range范围中的坐标。
                对于global_range=(10,10), local_range=(2,10)来说，只有(2,10)范围内的元素被置为2.0f, 而global_id的范围是(0, 0)->(3, 10), 即所有的work-item的global_id只可能是*/
                // temp_acc[wg_idx][wg_idy] += 1.0f;
                // atomic_ref<float, memory_order::relaxed, memory_scope::device> temp_atomic(temp_acc[wg_idx][wg_idy]);
                // temp_atomic.fetch_add(1.0f);
                // temp_acc[local_idx][local_idy] += 2.0f;
                // 使用原子加法更新 temp_acc
                // atomic_ref<float, memory_order::relaxed, memory_scope::device> temp_atomic(temp_acc[local_idx][local_idy]);
                // temp_atomic.fetch_add(1.0f);

                // int block_idx_Br = global_idx[0] * Br;
                // int block_idx_Bc = global_idx[1] * Bc;
                // auto& subQ_acc = subQ_acc_list[global_idx[0]];
                // auto& subK_acc = subK_acc_list[global_idx[1]];
                // auto& subV_acc = subV_acc_list[global_idx[1]];

                // 加载数据到 Joint Matrix
                // joint_matrix_load(subQ_acc, Q_buf, 0, block_idx_Br);
                // joint_matrix_load(subK, subK_acc, 0, block_idx_Bc);
                // // 执行矩阵乘法
                // subO = joint_matrix_mad(subQ, subK, subO);
                // // 存储结果
                // joint_matrix_store(subO, O_acc, block_idx_Br, block_idx_Bc);
            });
        }).wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
        std::terminate();
    }

    // 输出结果
    std::cout << "Output temp: ";
    for (int i = 0; i < temp.size(); ++i) {
        if (i % 10 == 0) {
            std::cout << std::endl;
        }
        std::cout << temp[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

// void flash_attn_sycl(
//     torch::Tensor Q,
//     torch::Tensor K,
//     torch::Tensor V,
//     torch::Tensor O,
//     torch::Tensor m,
//     torch::Tensor l,
//     int Br,
//     int Bc,
//     torch::Tensor mask) {

//     auto Q_data = Q.data_ptr<float>();
//     auto K_data = K.data_ptr<float>();
//     auto V_data = V.data_ptr<float>();
//     auto O_data = O.data_ptr<float>();
//     auto m_data = m.data_ptr<float>();
//     auto l_data = l.data_ptr<float>();
//     auto mask_data = mask.defined() ? mask.data_ptr<float>() : nullptr;

//     int batch_size = Q.size(0);
//     int seq_len = Q.size(1);
//     int d_model = Q.size(2);

//     queue q;

//     q.submit([&](handler& h) {
//         h.parallel_for(range<3>(batch_size, seq_len / Br, seq_len / Bc), [=](id<3> idx) {
//             int batch_idx = idx[0];
//             int row_start = idx[1] * Br;
//             int block_start_Bc = idx[2] * Bc;

//             if (row_start < seq_len && block_start_Bc < seq_len) {
//                 for (int i = 0; i < Br; ++i) {
//                     for (int j = 0; j < Bc; ++j) {
//                         int q_idx = batch_idx * seq_len * d_model + (row_start + i) * d_model;
//                         int k_idx = batch_idx * seq_len * d_model + (block_start_Bc + j) * d_model;
//                         int v_idx = k_idx;

//                         float Sij = 0.0;
//                         for (int d = 0; d < d_model; ++d) {
//                             Sij += Q_data[q_idx + d] * K_data[k_idx + d];
//                         }

//                         if (mask_data != nullptr) {
//                             int mask_idx = batch_idx * seq_len * seq_len + (row_start + i) * seq_len + (block_start_Bc + j);
//                             if (mask_data[mask_idx] == 0) {
//                                 Sij = NEG_INF;
//                             }
//                         }

//                         int m_idx = batch_idx * seq_len + row_start + i;
//                         atomic_fetch_max(&m_data[m_idx], Sij);

//                         int l_idx = batch_idx * seq_len + row_start + i;
//                         atomic_fetch_add(&l_data[l_idx], exp(Sij - m_data[m_idx]));

//                         int o_idx = batch_idx * seq_len * d_model + (row_start + i) * d_model;
//                         for (int d = 0; d < d_model; ++d) {
//                             atomic_fetch_add(&O_data[o_idx + d], exp(Sij - m_data[m_idx]) * V_data[v_idx + d]);
//                         }
//                     }
//                 }
//             }
//         });
//     }).wait();
// }