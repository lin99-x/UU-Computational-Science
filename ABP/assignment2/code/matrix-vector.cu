#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cublas_v2.h>

const int block_size = 128;

__global__ void matvec_simple(int M, int N, float *A, float *x, float *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < M)
    {
        float sum = 0.f;
        for (int j=0; j<N; j++)
        {
            sum += A[i + M * j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void matvec_transpose(int M, int N, float *A, float *x, float *y)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    float sum = 0.f;
    for (int j = 0; j < M; j++)
    {
      sum += A[i * N + j] * x[j];
    } 
    y[i] = sum;
  }
}


__global__ void set_vector(const int N, const float val, float *x)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N)
  {
    x[idx] = val;
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

// benchmark function matrix vector
void benchmark_matvec(const std::size_t M, const std::size_t N)
{
  float *A, *x, *y;

  // allocate memory on the device
  cudaMalloc(&A, M * N * sizeof(float));
  cudaMalloc(&x, N * sizeof(float));
  cudaMalloc(&y, M * sizeof(float));

  const int n_blocks = 1 + (M - 1) / block_size;
  dim3 grid_dim(n_blocks, n_blocks);
  dim3 block_dim(block_size, block_size);

  // cudaMemcpy(A, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);
  set_vector<<<n_blocks, block_size>>>(N, 17.f, x);
  set_matrix<<<n_blocks, block_size>>>(M, N, 10.f, A);
  set_vector<<<n_blocks, block_size>>>(M, 0.f, y);

  std::vector<float> result_host(M);

  const int n_tests = 20;
  const int n_repeat = 500;
  double best = 1e10, worst = 0, avg = 0;
  for (int t = 0; t < n_tests; ++t)
  {
    const auto t1 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < n_repeat; ++rep)
    {
      matvec_simple<<<n_blocks, block_size>>>(M, N, A, x, y);
    }
    cudaDeviceSynchronize();
    const double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();
    best = std::min(best, time / n_repeat);
    worst = std::max(worst, time / n_repeat);
    avg += time / n_repeat;
  }

  cudaMemcpy(result_host.data(), y, M * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A);
  cudaFree(x);
  cudaFree(y);

  std::cout << "matrix size " << std::setw(8) << M << " " << N
          << " : min/avg/max: " << std::setw(11) << best << " "
          << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
          << " seconds or " << std::setw(8) << 1e-6 * 2 * M * N / best
          << " MUPD/s or " << std::setw(8)
          << 1e-9 * sizeof(float) * (M * N + M + N) / best << " GB/s" << std::endl;
}

// benchmark function for cuBLAS matrix vector
void benchmark_cuBLAS(const std::size_t M, const std::size_t N)
{
  float *A, *x, *y;

  // allocate memory on the device
  cudaMalloc(&A, M * N * sizeof(float));
  cudaMalloc(&x, N * sizeof(float));
  cudaMalloc(&y, M * sizeof(float));

  const int n_blocks = 1 + (M - 1) / block_size;
  dim3 grid_dim(n_blocks, n_blocks);
  dim3 block_dim(block_size, block_size);

  // cudaMemcpy(A, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);
  set_vector<<<n_blocks, block_size>>>(N, 17.f, x);
  set_matrix<<<n_blocks, block_size>>>(M, N, 10.f, A);
  set_vector<<<n_blocks, block_size>>>(M, 0.f, y);

  std::vector<float> result_host(M);

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cout << "CUBLAS initialization failed\n";
    std::abort();
  }
  float alpha = 1.f;
  float beta = 0.;

  const int n_tests = 20;
  const int n_repeat = 500;
  double best = 1e10, worst = 0, avg = 0;
  for (int t=0; t<n_tests; ++t)
  {
    const auto t1 = std::chrono::steady_clock::now();
    for (int rep=0; rep<n_repeat; ++rep)
    {
      stat = cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, A, M, x, 1, &beta, y, 1);
      if (stat != CUBLAS_STATUS_SUCCESS)
      {
        std::cout << "CUBLAS operation failed\n";
        std::abort();
      }
    }
    cudaDeviceSynchronize();
    const double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();
    best  = std::min(best, time / n_repeat);
    worst = std::max(worst, time / n_repeat);
    avg += time / n_repeat;
  }

  cudaMemcpy(result_host.data(), y, M * sizeof(float), cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(A);
  cudaFree(x);
  cudaFree(y);

  std::cout << "matrix size " << std::setw(8) << M << " " << N
          << " : min/avg/max: " << std::setw(11) << best << " "
          << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
          << " seconds or " << std::setw(8) << 1e-6 * 2 * M * N / best
          << " MUPD/s or " << std::setw(8)
          << 1e-9 * sizeof(float) * (M * N + M + N) / best << " GB/s" << std::endl;
}


int main(int argc, char **argv)
{
  // if (argc % 2 == 0)
  // {
  //   std::cout << "Error, expected odd number of common line arguments"
  //             << std::endl
  //             << "Expected line of the form" << std::endl
  //             << "-min 100 -max 10000" << std::endl;
  //   std::abort();
  // }

  int N_min = 10;
  int N_max = 1000;

  // for (int l = 1; l < argc; l += 2)
  //   {
  //     std::string option = argv[l];
  //     if (option == "-min")
  //       N_min = static_cast<long>(std::stod(argv[l + 1]));
  //     else if (option == "-max")
  //       N_max = static_cast<long>(std::stod(argv[l + 1]));
  //     else
  //       std::cout << "Unknown option " << option << " - ignored!" << std::endl;
  //   }

  // std::cout << "Benchmarking matrix-vector product..." << std::endl;

  for (int i = N_min; i <= N_max; i = i + (N_max / 30))
  {
    i = ceil(i);
    benchmark_cuBLAS(16384, i);
  }
  // benchmark_matvec(5000, 5000);

  return 0;
}