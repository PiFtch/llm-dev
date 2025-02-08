#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    range<2> global_range(16, 16);
    range<2> local_range(1, 1);

    auto devices = device::get_devices();
    std::cout << "Available SYCL devices:" << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  " << dev.get_info<info::device::name>() << std::endl;
    }

    queue q;
    try {
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<2>(global_range, local_range), [=](nd_item<2> item) {
                sub_group sg = item.get_sub_group();
                joint_matrix<sub_group, half, use::a, 2, 2, layout::row_major> tA;
            });
        }).wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
        std::terminate();
    }

    return 0;
}