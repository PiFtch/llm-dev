#include <CL/sycl.hpp>
// #include <torch/extension.h>

using namespace sycl;

// torch::Tensor custom_sycl_op(torch::Tensor input) {
//     auto output = torch::zeros_like(input);

//     queue q;
//     auto input_data = input.data_ptr<float>();
//     auto output_data = output.data_ptr<float>();

//     q.submit([&](handler& h) {
//         h.parallel_for(range<1>(input.size(0)), [=](id<1> i) {
//             output_data[i] = input_data[i] * 2.0f + 3.0f;
//         });
//     }).wait();

//     return output;
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("custom_sycl_op", &custom_sycl_op, "Custom SYCL Op");
// }

void custom_sycl_op(int** input) {
    auto output = torch::zeros_like(input);

    queue q;
    auto input_data = input.data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    q.submit([&](handler& h) {
        h.parallel_for(range<1>(input.size(0)), [=](id<1> i) {
            output_data[i] = input_data[i] * 2.0f + 3.0f;
        });
    }).wait();

    // return output;
}