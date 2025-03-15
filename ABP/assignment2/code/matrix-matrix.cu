#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cublas_v2.h>

const int block_size = 32;

__global__ void matmul_simple(int M, int N, int K, float *A, float *B, float *C)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < M && col < K)
  {
    float sum = 0.f;
    for (int j = 0; j < N; j++)
    {
      sum += A[row + j * M] * B[j + col * N];
    }
    C[row + col * M] = sum;
  }
}


__global__ void set_matrix(const int M, const int N, const float val, float *A)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int idy = threadIdx.y + blockIdx.y * blockDim.y;
  if (idx < M && idy < N)
  {
    A[idx + idy * M] = val;
  }
}


// benchmark function matrix matrix
void benchmark_matmul(const std::size_t M, const std::size_t N, const std::size_t K)
{
  float *A, *B, *C;

  // allocate memory on the device
  cudaMalloc(&A, M * N * sizeof(float));
  cudaMalloc(&B, N * K * sizeof(float));
  cudaMalloc(&C, M * K * sizeof(float));

  const int n_blocks = 1 + (M - 1) / block_size;
  dim3 grid_dim(n_blocks, n_blocks);
  dim3 block_dim(block_size, block_size);

  // cudaMemcpy(A, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);
  set_matrix<<<grid_dim, block_dim>>>(M, N, 10.f, A);
  set_matrix<<<grid_dim, block_dim>>>(N, K, 17.f, B);
  set_matrix<<<grid_dim, block_dim>>>(M, K, 0.f, C);

  // std::vector<float> result_host(M * K);

  const int n_tests = 20;
  const int n_repeat = 50;
  double best = 1e10, worst = 0, avg = 0;
  for (int t = 0; t < n_tests; ++t)
  {
    const auto t1 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < n_repeat; ++rep)
    {
      matmul_simple<<<grid_dim, block_dim>>>(M, N, K, A, B, C);
    }
    cudaDeviceSynchronize();
    const double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();
    best = std::min(best, time / n_repeat);
    worst = std::max(worst, time / n_repeat);
    avg += time / n_repeat;
  }
  avg = avg / n_tests;
  // cudaMemcpy(result_host.data(), y, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  // for()
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  std::cout << "matrix size " << std::setw(8) << M << " " << N
          << " : min/avg/max: " << std::setw(11) << best << " "
          << std::setw(11) << avg << " " << std::setw(11) << worst
          << " seconds or " << std::setw(8) << 1e-9 * 2 * M * N * K/ avg
          << " GFlop/s or " << std::setw(8)
          << 1e-9 * sizeof(float) * (M * N + N * K + N * K) / best << " GB/s" << std::endl;
}


// benchmark function matrix matrix
void benchmark_matmul_cublas(const std::size_t M, const std::size_t N, const std::size_t K)
{
  float *A, *B, *C;

  // allocate memory on the device
  cudaMalloc(&A, M * N * sizeof(float));
  cudaMalloc(&B, N * K * sizeof(float));
  cudaMalloc(&C, M * K * sizeof(float));

  const int n_blocks = 1 + (M - 1) / block_size;
  dim3 grid_dim(n_blocks, n_blocks);
  dim3 block_dim(block_size, block_size);

  // cudaMemcpy(A, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);
  set_matrix<<<grid_dim, block_dim>>>(M, N, 10.f, A);
  set_matrix<<<grid_dim, block_dim>>>(N, K, 17.f, B);
  set_matrix<<<grid_dim, block_dim>>>(M, K, 0.f, C);


    cublasHandle_t handle;
           cublasStatus_t stat = cublasCreate(&handle);
           if (stat != CUBLAS_STATUS_SUCCESS)
        {
              std::cout << "CUBLAS initialization failed\n";
              std::abort();
        }

    float alpha = 1.f;
    float beta = 0.;

    const unsigned int n_tests = 20;
    const unsigned int n_repeat = 50; // Number of times to repeat the kernel execution
    double best = 1e10, worst = 0, avg = 0;
   
    for (unsigned int t = 0; t < n_tests; ++t)
    {
        const auto t1 = std::chrono::steady_clock::now();
        
        for (unsigned int rep = 0; rep < n_repeat; ++rep)
         {
        stat =cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,&alpha, A, N, B, N, &beta, C, N);
        if (stat != CUBLAS_STATUS_SUCCESS)
          {
              std::cout << "CUBLAS operation failed\n";
              std::abort();
          }
        }
        
        cudaDeviceSynchronize();

        const double time = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t1).count() / n_repeat; // Average time per repetition

        best = std::min(best, time);
        worst = std::max(worst, time);
        avg += time;
    } 

    avg /= n_tests;

    double FLOPs = 2.0 * M * N * K;
    double GFLOPs = FLOPs / (avg * 1e9);
    

    std::cout << "Size " << M << "x" << N << "x" << K 
              << ", Time: " << avg << " s"
              << ", Best: " << best << " s"
              << ", Worst: " << worst << " s"
              << ", Performance: " << GFLOPs << " GFLOP/s" << std::endl;


    // Cleanup
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}


int main(int argc, char **argv)
{
  int N_min = 100;
  int N_max = 5000;

  std::cout << "Benchmarking matrix-matrix my product..." << std::endl;

  for (int i = N_min; i <= N_max; i = i + (N_max / 20))
  {
    i = ceil(i);
    benchmark_matmul_cublas(i, i, i);
  }

  return 0;
}