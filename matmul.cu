#include <cuda_runtime.h>
#include <stdio.h>


// __global__ void Matmul_kernel_naive(float *M_d, float *N_d, float *P_d, int width)
// {
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     float Pvalue = 0;

//     for (int k = 0; k < width; k++)
//     {
//         float Melement = M_d[ty * width + k];
//         float Nelement = N_d[k * width + tx];
//         Pvalue += Melement * Nelement;
//     }

//     P_d[ty * width + tx];
// }

#define WIDTH (8192)   // 4x4 matrix
#define TILE_WIDTH (32) // 矩阵分为相等大小的block，每个block里每一行/列元素的数量（方阵）
__global__ void Matmul_kernel_block(float *M_d, float *N_d, float *P_d, int width)
{
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // 行优先布局的元素行号
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Pvalue = 0;

    for (int k = 0; k < width; k++)
    {
        Pvalue += M_d[row * width + k] * N_d[k * width + col];
    }

    P_d[row * width + col] = Pvalue;
}

int main()
{
    cudaError_t cuError;

    int size = WIDTH * WIDTH * sizeof(float);

    float *M_d = NULL;
    float *N_d = NULL;
    float *P_d = NULL;

    cuError = cudaMalloc((void**)&M_d, size);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMalloc()\n");
        exit(1);
    }

    cuError = cudaMalloc((void**)&N_d, size);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMalloc()\n");
        exit(1);
    }

    cuError = cudaMalloc((void**)&P_d, size);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMalloc()\n");
        exit(1);
    }

    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            M_h[i * WIDTH + j] = (float)rand() / RAND_MAX;
            N_h[i * WIDTH + j] = (float)rand() / RAND_MAX;

            // printf("%.8f ", M_h[i * WIDTH + j]);
        }
    }

    cuError = cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMemcpy()\n");
        exit(1);
    }
    cuError = cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMemcpy()\n");
        exit(1);
    }

    // dim3 dimGrid(2, 2, 1);
    // dim3 dimBlock(4, 2, 2);

    // Matmul_kernel_naive<<<dimGrid, dimBlock>>>(M_d, N_d, P_d);

    cudaStream_t stream[4];
    for (int i = 0; i < 4; i++)
    cudaStreamCreate(&stream[i]);

    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    for (int i = 0; i < 4; i++)
    {
        Matmul_kernel_block<<<dimGrid, dimBlock, 0, stream[i]>>>(M_d, N_d, P_d, WIDTH);
        cuError = cudaGetLastError();
        if (cuError != cudaSuccess)
        {
            printf("Failed to launch kernel, %s\n", cudaGetErrorString(cuError));
            exit(1);
        }
    }

    cuError = cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    if (cuError != cudaSuccess)
    {
        printf("Failed to cudaMemcpy()\n");
        exit(1);
    }

    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            printf("%.8f ", P_h[i * WIDTH + j]);
        }
        printf("\n");
    }

    return 0;
}