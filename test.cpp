#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    queue q;
    q.submit([&](handler &h) {
        h.parallel_for<class nd_item_kernel>(nd_range<1>(range<1>(16), range<1>(4)), [=](nd_item<1> item) {
            // Get the global ID
            auto global_id = item.get_global_id(0);
            // Get the local ID
            auto local_id = item.get_local_id(0);
            printf("Global ID: %d, Local ID: %d\n", global_id, local_id);
        });
    }).wait();
    return 0;
}
