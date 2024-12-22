#include <cuda_runtime.h>
#include <stdio.h>

#define N 500000
__global__ void shortKernel(float *out_d, float *in_d, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num)
    {
        out_d[idx] = 1.23 * in_d[idx];
        // out_d[idx] = 1.0f;
    }
}

#define NSTEP 10
#define NKERNEL 20


void graph_launch()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool graphCreated = false;
    cudaGraph_t cuGraph;
    cudaGraphExec_t cuGraphInstance;

    int size = N * sizeof(float);
    cudaError_t cuError;
    float *in_d;
    float *out_d;

    float *in_h = (float*)malloc(size);
    float *out_h = (float*)malloc(size);

    for (int i = 0; i < N; i++)
    {
        in_h[i] = rand() / (float)RAND_MAX;
    }

    cuError = cudaMalloc((void**)&in_d, size);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    cuError = cudaMalloc((void**)&out_d, size);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    cuError = cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int istep = 0; istep < NSTEP; istep++)
    {
        if (!graphCreated)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
            {
                shortKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(out_d, in_d, N);
            }
            cudaStreamEndCapture(stream, &cuGraph);
            cudaGraphInstantiate(&cuGraphInstance, cuGraph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(cuGraphInstance, stream);
        cudaStreamSynchronize(stream);
    }
}

int main()
{
    graph_launch();
    return 0;
    int size = N * sizeof(float);
    cudaError_t cuError;
    float *in_d;
    float *out_d;

    float *in_h = (float*)malloc(size);
    float *out_h = (float*)malloc(size);

    for (int i = 0; i < N; i++)
    {
        in_h[i] = rand() / (float)RAND_MAX;
    }

    cuError = cudaMalloc((void**)&in_d, size);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    cuError = cudaMalloc((void**)&out_d, size);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    cuError = cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to malloc (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // cudaStream_t streams[14];
    // for (int i = 0; i < 14; i++)
    // {
    //     cudaStreamCreate(&streams[i]);
    // }
    

    // int blocks = 256;
    // int threads = 16;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    for (int istep = 0; istep < NSTEP; istep++)
    {
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++)
        {
            shortKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(out_d, in_d, N);
            // cudaStreamSynchronize(stream);
        }
        cudaStreamSynchronize(stream);
    }

    cuError = cudaGetLastError();
    if (cuError != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(cuError));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f ", out_h[i]);
    // }
    // printf("\n");
    
    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}