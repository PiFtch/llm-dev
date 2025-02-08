#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
// #include <sycl/ext/oneapi/matrix/matrix-unified.hpp>
// #include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace sycl;
// using namespace sycl::ext::oneapi::experimental::matrix;
using namespace sycl::ext::intel::experimental::matrix;
// using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    const int M = 4;
    const int K = 4;
    const int N = 4;
    const int tM = 2;
    const int tK = 2;
    const int tN = 2;
    const int SG_SIZE = 2;

    int8_t *memA = new int8_t[M * K];
    int8_t *memB = new int8_t[K * N];
    int8_t *memC = new int8_t[M * N];

    auto devices = device::get_devices();
    // 列出所有可用设备
    std::cout << "Available SYCL devices:" << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  " << dev.get_info<info::device::name>() << std::endl;
    }

    queue q(devices[2]);
    range<2> G = {M/tM, N};
    range<2> L = {1, SG_SIZE};

    try {
        auto bufA = sycl::buffer{memA, sycl::range{M*K}};
        auto bufB = sycl::buffer{memB, sycl::range{K*N}};
        auto bufC = sycl::buffer{memC, sycl::range{M*N}};
        q.submit([&](sycl::handler& cgh) {
            auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
            auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
            auto accC = sycl::accessor{bufC, cgh, sycl::read_write};

            cgh.parallel_for(nd_range<2>(G, L), [=](nd_item<2> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto global_idx = item.get_global_id(0);
                const auto global_idy = item.get_global_id(1);
                const auto sg_startx = global_idx - item.get_local_id(0);
                const auto sg_starty = global_idy - item.get_local_id(1);
                sub_group sg = item.get_sub_group();
                joint_matrix<sub_group, int8_t, use::a, tM, tK, layout::row_major> tA;
                joint_matrix<sub_group, int8_t, use::b, tK, tN, layout::ext_intel_packed> tB;
                joint_matrix<sub_group, int8_t, use::accumulator, tM, tN> tC;
                joint_matrix_fill(sg, tC, 0);
                for (int k = 0; k < K; k += tK) {
                    // joint_matrix_load(sg, tA, accA.template get_multi_ptr<sycl::access::decorated::no>() + sg_startx * tM * K + k, K);
                    joint_matrix_load(sg, tA, accA.get_multi_ptr<access::decorated::no>() + sg_startx * tM * K + k, K);
                    joint_matrix_load(sg, tB, accB.template get_multi_ptr<sycl::access::decorated::no>() + k * N*4 + sg_starty/SG_SIZE*tN*4, N*4);
                    joint_matrix_mad(sg, tC, tA, tB, tC);
                }

                // joint_matrix_apply(sg, tC, [=](int8_t x) {
                //     x *= alpha;
                // });

                joint_matrix_store(sg, tC, accC.template get_multi_ptr<sycl::access::decorated::no>() + sg_startx * tM * N + sg_starty/SG_SIZE*tN, N, layout::row_major);
            });
        });
        q.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

}