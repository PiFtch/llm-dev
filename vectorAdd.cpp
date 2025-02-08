#include <sycl/sycl.hpp>
// #include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace sycl;

int main() {
    // select device
    auto devices = device::get_devices();
    // 列出所有可用设备
    std::cout << "Available SYCL devices:" << std::endl;
    for (const auto& dev : devices) {
        std::cout << "  " << dev.get_info<info::device::name>() << std::endl;
    }

    // 定义向量大小
    const int N = 1024;

    // 初始化输入向量
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    // 创建SYCL队列
    queue q(devices[2]);
    {
        // 创建SYCL缓冲区
        buffer<float, 1> A_buf(A.data(), range<1>(N));
        buffer<float, 1> B_buf(B.data(), range<1>(N));
        buffer<float, 1> C_buf(C.data(), range<1>(N));

        // 提交命令组到队列
        q.submit([&](handler& h) {
            // 获取缓冲区访问权限
            auto A_acc = A_buf.get_access<access::mode::read>(h);
            auto B_acc = B_buf.get_access<access::mode::read>(h);
            auto C_acc = C_buf.get_access<access::mode::write>(h);

            // 执行并行向量加法
            h.parallel_for(range<1>(N), [=](id<1> i) {
                C_acc[i] = A_acc[i] + B_acc[i];
            });
        }).wait();
    }
    // 输出结果
    std::cout << "Result: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}