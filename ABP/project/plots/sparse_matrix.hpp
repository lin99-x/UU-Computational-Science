
#ifndef sparse_matrix_hpp
#define sparse_matrix_hpp

#include <utility>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif

#include <omp.h>

#include <vector>

#include "vector.hpp"


#ifndef DISABLE_CUDA
template <typename Number>
__global__ void compute_spmv(const std::size_t N,
                             const std::size_t *row_starts,
                             const unsigned int *column_indices,
                             const Number *values,
                             const Number *x,
                             Number *y)
{
  // TODO implement for GPU
  const int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < N)
  {
    Number sum = 0;
    // loop over all entries in row
    for (std::size_t idx = row_starts[row]; idx < row_starts[row + 1]; idx++)
    {
      sum += values[idx] * x[column_indices[idx]];
    }
    y[row] = sum;
  }
}

template <typename Number>
__global__ void compute_spMVM(std::size_t n_rows,
                              unsigned int chunk_size,
                              unsigned int *column_indices_sell_c,
                              unsigned int *cl,
                              unsigned int *cs,
                              Number *values_sell_c,
                              const Number *x,
                              Number *y)
{
  // implement for GPU, Sell-C-sigma
  const int tid = threadIdx.x;
  const int i = blockIdx.x;         // chunk index

  Number sum = 0;
  for (int j = 0; j < cl[i]; j++)
  {
    sum += values_sell_c[cs[i] + j * chunk_size + tid] * x[column_indices_sell_c[cs[i] + j * chunk_size + tid]];
  }
  y[i * chunk_size + tid] = sum;
}
#endif



// Sparse matrix in compressed row storage (crs) format

template <typename Number>
class SparseMatrix
{
public:
  static const int block_size = Vector<Number>::block_size;

  SparseMatrix(const std::vector<unsigned int> &row_lengths,
               const MemorySpace                memory_space,
               const MPI_Comm                   communicator)
    : communicator(communicator),
      memory_space(memory_space)
  {
    n_rows     = row_lengths.size();
    row_starts = new std::size_t[n_rows + 1];

    // for sell-c-sigma structure
    int total_size = 0;
    std::vector<unsigned int> max_row_length_chunk(n_rows / chunk_size);
    // go through each chunk
    for (unsigned int i = 0; i < n_rows / chunk_size; ++i) {
      max_row_length_chunk[i] = 0;
      // go through each row in chunk
      for (unsigned int j = 0; j < chunk_size; ++j) {
        if (row_lengths[i * chunk_size + j] > max_row_length_chunk[i]) {
          max_row_length_chunk[i] = row_lengths[i * chunk_size + j];
        }
      }
      total_size += chunk_size * max_row_length_chunk[i];
    }

#pragma omp parallel for
    for (unsigned int row = 0; row < n_rows + 1; ++row)
      row_starts[row] = 0;

    for (unsigned int row = 0; row < n_rows; ++row)
      row_starts[row + 1] = row_starts[row] + row_lengths[row];

    const std::size_t n_entries = row_starts[n_rows];
    

    if (memory_space == MemorySpace::CUDA)
      {
        std::size_t *host_row_starts = row_starts;
        row_starts = 0;
        AssertCuda(cudaMalloc(&row_starts, (n_rows + 1) * sizeof(std::size_t)));
        AssertCuda(cudaMemcpy(row_starts,
                              host_row_starts,
                              (n_rows + 1) * sizeof(std::size_t),
                              cudaMemcpyHostToDevice));
        delete[] host_row_starts;

        AssertCuda(cudaMalloc(&column_indices,
                              n_entries * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));

        // for sell-c-sigma structure
        AssertCuda(cudaMalloc(&column_indices_sell_c, total_size * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&cl, (n_rows / chunk_size) * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&cs, (n_rows / chunk_size) * sizeof(unsigned int)));
        AssertCuda(cudaMalloc(&values_sell_c, total_size * sizeof(Number)));
        cudaDeviceSynchronize();
        AssertCuda(cudaPeekAtLastError());

#ifndef DISABLE_CUDA
        const unsigned int n_blocks =
          (n_entries + block_size - 1) / block_size;
        set_entries<<<n_blocks, block_size>>>(n_entries, 0U, column_indices);   // initial it to zeros
        set_entries<<<n_blocks, block_size>>>(n_entries, Number(0), values);    // initial it to zeros
        AssertCuda(cudaPeekAtLastError());
#endif
      }
    else
      {
        column_indices = new unsigned int[n_entries];
        values         = new Number[n_entries];

        // for sell-c-sigma structure, just for test
        column_indices_sell_c = new unsigned int[total_size];
        cl = new unsigned int[n_rows / chunk_size];
        cs = new unsigned int[n_rows / chunk_size];
        values_sell_c = new Number[total_size];

#pragma omp parallel for
        for (std::size_t i = 0; i < n_entries; ++i)
          column_indices[i] = 0;

#pragma omp parallel for
        for (std::size_t i = 0; i < n_entries; ++i)
          values[i] = 0;
      }

    n_global_nonzero_entries = mpi_sum(n_entries, communicator);
  }

  ~SparseMatrix()
  {
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        cudaFree(row_starts);
        cudaFree(column_indices);
        cudaFree(values);

        // for sell-c-sigma structure
        cudaFree(column_indices_sell_c);
        cudaFree(cl);
        cudaFree(cs);
        cudaFree(values_sell_c);
#endif
      }
    else
      {
        delete[] row_starts;
        delete[] column_indices;
        delete[] values;

        // // for sell-c-sigma structure
        // delete[] column_indices_sell_c;
        // delete[] cl;
        // delete[] cs;
        // delete[] values_sell_c;
      }
  }

  SparseMatrix(const SparseMatrix &other)
    : communicator(other.communicator),
      memory_space(other.memory_space),
      n_rows(other.n_rows),
      n_global_nonzero_entries(other.n_global_nonzero_entries)
  {
    if (memory_space == MemorySpace::CUDA)
      {
        AssertCuda(cudaMalloc(&row_starts, (n_rows + 1) * sizeof(std::size_t)));
        AssertCuda(cudaMemcpy(row_starts,
                              other.row_starts,
                              (n_rows + 1) * sizeof(std::size_t),
                              cudaMemcpyDeviceToDevice));

        std::size_t n_entries = 0;
        AssertCuda(cudaMemcpy(&n_entries,
                              other.row_starts + n_rows,        // n_entries is stored in the last element of the row_starts array
                              sizeof(std::size_t),
                              cudaMemcpyDeviceToHost));
        AssertCuda(cudaMalloc(&column_indices,
                              n_entries * sizeof(unsigned int)));
        AssertCuda(cudaMemcpy(column_indices,
                              other.column_indices,
                              n_entries * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

        AssertCuda(cudaMalloc(&values, n_entries * sizeof(Number)));
        AssertCuda(cudaMemcpy(values,
                              other.values,
                              n_entries * sizeof(Number),
                              cudaMemcpyDeviceToDevice));
      }
    else
      {

      }
  }

  // do not allow copying matrix
  SparseMatrix operator=(const SparseMatrix &other) = delete;

  unsigned int m() const
  {
    return n_rows;
  }

  std::size_t n_nonzero_entries() const
  {
    return n_global_nonzero_entries;
  }

  void add_row(unsigned int               row,
               std::vector<unsigned int> &columns_of_row,
               std::vector<Number> &      values_in_row)
  {
    if (columns_of_row.size() != values_in_row.size())
      {
        std::cout << "column_indices and values must have the same size!"
                  << std::endl;
        std::abort();
      }
    for (unsigned int i = 0; i < columns_of_row.size(); ++i)
      {
        column_indices[row_starts[row] + i] = columns_of_row[i];
        values[row_starts[row] + i]         = values_in_row[i];
      }
  }

  void allocate_ghost_data_memory(const std::size_t n_ghost_entries)
  {
    ghost_entries.clear();
    ghost_entries.reserve(n_ghost_entries);
#pragma omp parallel for
    for (unsigned int i = 0; i < n_ghost_entries; ++i)
      {
        ghost_entries[i].index_within_result         = 0;
        ghost_entries[i].index_within_offproc_vector = 0;
        ghost_entries[i].value                       = 0.;
      }
  }

  void add_ghost_entry(const unsigned int local_row,
                       const unsigned int offproc_column,
                       const Number       value)
  {
    GhostEntryCoordinateFormat entry;
    entry.value                       = value;
    entry.index_within_result         = local_row;
    entry.index_within_offproc_vector = offproc_column;
    ghost_entries.push_back(entry);
  }

  // In real codes, the data structure we pass in manually here could be
  // deduced from the global indices that are accessed. In the most general
  // case, it takes some two-phase index lookup via a dictionary to find the
  // owner of particular columns (sometimes called consensus algorithm).
  void set_send_and_receive_information(
    std::vector<std::pair<unsigned int, std::vector<unsigned int>>>
                                                       send_indices,
    std::vector<std::pair<unsigned int, unsigned int>> receive_indices)
  {
    this->send_indices    = send_indices;
    std::size_t send_size = 0;
    for (auto i : send_indices)
      send_size += i.second.size();
    send_data.resize(send_size);
    this->receive_indices    = receive_indices;
    std::size_t receive_size = 0;
    for (auto i : receive_indices)
      receive_size += i.second;
    receive_data.resize(receive_size);

    const unsigned int my_mpi_rank = get_my_mpi_rank(communicator);

    if (receive_size > ghost_entries.size())
      {
        std::cout << "Error, you requested exchange of more entries than what "
                  << "there are ghost entries allocated in the matrix, which "
                  << "does not make sense. Check matrix setup." << std::endl;
        std::abort();
      }
  }


  void apply(const Vector<Number> &src, Vector<Number> &dst) const
  {
    if (m() != src.size_on_this_rank() || m() != dst.size_on_this_rank())
      {
        std::cout << "vector sizes of src " << src.size_on_this_rank()
                  << " and dst " << dst.size_on_this_rank()
                  << " do not match matrix size " << m() << std::endl;
        std::abort();
      }

#ifdef HAVE_MPI
    // start exchanging the off-processor data
    std::vector<MPI_Request> mpi_requests(send_indices.size() +
                                          receive_indices.size());
    for (unsigned int i = 0, count = 0; i < receive_indices.size();
         count += receive_indices[i].second, ++i)
      MPI_Irecv(receive_data.data() + count,
                receive_indices[i].second * sizeof(Number),
                MPI_BYTE,
                receive_indices[i].first,
                /* mpi_tag */ 29,
                communicator,
                &mpi_requests[i]);
    for (unsigned int i = 0, count = 0; i < send_indices.size(); ++i)
      {
#  pragma omp parallel for
        for (unsigned int j = 0; j < send_indices[i].second.size(); ++j)
          send_data[count + j] = src(send_indices[i].second[j]);

        MPI_Isend(send_data.data() + count,
                  send_indices[i].second.size() * sizeof(Number),
                  MPI_BYTE,
                  send_indices[i].first,
                  /* mpi_tag */ 29,
                  communicator,
                  &mpi_requests[i + receive_indices.size()]);
        count += send_indices[i].second.size();
      }
#endif

    // main loop for the sparse matrix-vector product
    if (memory_space == MemorySpace::CUDA)
      {
#ifndef DISABLE_CUDA
        // TODO implement for GPU (with CRS and ELLPACK/SELL-C-sigma)
        AssertCuda(cudaPeekAtLastError());
        const int n_blocks = (n_rows + block_size - 1) / block_size;
        compute_spmv<Number><<<n_blocks, block_size>>>(n_rows, row_starts, column_indices, values, src.begin(), dst.begin());
#endif
      }
    else
      {
#pragma omp parallel for
        for (unsigned int row = 0; row < n_rows; ++row)
          {
            Number sum = 0;
            for (std::size_t idx = row_starts[row]; idx < row_starts[row + 1];
                 ++idx)
              sum += values[idx] * src(column_indices[idx]);
            dst(row) = sum;
          }
      }

#ifdef HAVE_MPI
    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);

    // work on the off-processor data. do not do it in parallel because we do
    // not know whether two parts would work on the same entry of the result
    // vector
    for (auto &entry : ghost_entries)
      dst(entry.index_within_result) +=
        entry.value * receive_data[entry.index_within_offproc_vector];
#endif
  }

  void copy_data_to_device_sellc(){
    int total_size = 0;
    for (unsigned int i = 0; i < n_rows / chunk_size; ++i){
      total_size += cl[i] * chunk_size;
    }

    AssertCuda(cudaMalloc(&column_indices_sellc_cuda, total_size * sizeof(unsigned int)));
    AssertCuda(cudaMalloc(&values_sellc_cuda, total_size * sizeof(Number)));
    AssertCuda(cudaMalloc(&cs_cuda, (n_rows / chunk_size) * sizeof(unsigned int)));
    AssertCuda(cudaMalloc(&cl_cuda, (n_rows / chunk_size) * sizeof(unsigned int)));

    // copy data to device
    AssertCuda(cudaMemcpy(column_indices_sellc_cuda, column_indices_sell_c, total_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(values_sellc_cuda, values_sell_c, total_size * sizeof(Number), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(cs_cuda, cs, (n_rows / chunk_size) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    AssertCuda(cudaMemcpy(cl_cuda, cl, (n_rows / chunk_size) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    AssertCuda(cudaPeekAtLastError());
  }

  void apply_sell_c(const Vector<Number> &src, Vector<Number> &dst, int cuda_flag) const
  {
    if (m() != src.size_on_this_rank() || m() != dst.size_on_this_rank())
      {
        std::cout << "vector sizes of src " << src.size_on_this_rank()
                  << " and dst " << dst.size_on_this_rank()
                  << " do not match matrix size " << m() << std::endl;
        std::abort();
      }
    
    if (cuda_flag)
      {
        const unsigned int n_blocks = n_rows / chunk_size;
        // C = 32
        dim3 grid_dim(n_blocks, 1);
        dim3 block_dim(32, 1);
        compute_spMVM<Number><<<grid_dim, block_dim>>>(n_rows, chunk_size, column_indices_sellc_cuda, cl_cuda, cs_cuda, values_sellc_cuda, src.begin(), dst.begin());
      }
    else
      {
        // for test
        for (int i=0; i<dst.size(); i++)
        {
          dst(i) = 0;
        }
        for (int i=0; i<n_rows/chunk_size; i++)
        {
          for (int j=0; j<cl[i]; j++)
          {
            for (int k=0; k<chunk_size; k++)
            {
              dst(i*chunk_size+k) += values_sell_c[cs[i]+j*chunk_size+k] * src(column_indices_sell_c[cs[i]+j*chunk_size+k]);
            }
          }
        }
      }
  }


  SparseMatrix copy_to_device()
  {
    if (memory_space == MemorySpace::CUDA)
      {
        std::cout << "Copy between device matrices not implemented"
                  << std::endl;
        exit(EXIT_FAILURE);
        // return dummy
        return SparseMatrix(std::vector<unsigned int>(),
                            MemorySpace::CUDA,
                            communicator);
      }
    else
      {
        std::vector<unsigned int> row_lengths(n_rows);
        for (unsigned int i = 0; i < n_rows; ++i)
          row_lengths[i] = row_starts[i + 1] - row_starts[i];


        // for sell-c-sigma structure
        int total_size = 0;
        std::vector<unsigned int> max_row_length_chunk(n_rows / chunk_size);
        // go through each chunk
        for (unsigned int i = 0; i < n_rows / chunk_size; ++i) {
          max_row_length_chunk[i] = 0;
          // go through each row in chunk
          for (unsigned int j = 0; j < chunk_size; ++j) {
            if (row_lengths[i * chunk_size + j] > max_row_length_chunk[i]) {
              max_row_length_chunk[i] = row_lengths[i * chunk_size + j];
            }
          }
          total_size += chunk_size * max_row_length_chunk[i];
        }

        SparseMatrix other(row_lengths,
                           MemorySpace::CUDA,
                           communicator);
        AssertCuda(cudaMemcpy(other.column_indices,
                              column_indices,
                              row_starts[n_rows] * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        AssertCuda(cudaMemcpy(other.values,
                              values,
                              row_starts[n_rows] * sizeof(Number),
                              cudaMemcpyHostToDevice));

        // // for sell-c structure
        // AssertCuda(cudaMemcpy(other.column_indices_sell_c,
        //                       column_indices_sell_c,
        //                       total_size * sizeof(unsigned int),
        //                       cudaMemcpyHostToDevice));
        // AssertCuda(cudaMemcpy(other.values_sell_c,
        //                       values_sell_c,
        //                       total_size * sizeof(Number),
        //                       cudaMemcpyHostToDevice));
        // AssertCuda(cudaMemcpy(other.cl,
        //                       cl,
        //                       (n_rows / chunk_size) * sizeof(unsigned int),
        //                       cudaMemcpyHostToDevice));
        // AssertCuda(cudaMemcpy(other.cs,
        //                       cs,
        //                       (n_rows / chunk_size) * sizeof(unsigned int),
        //                       cudaMemcpyHostToDevice));
        return other;
      }
  }

  std::size_t memory_consumption() const
  {
    return n_global_nonzero_entries * (sizeof(Number) + sizeof(unsigned int)) +
           (n_rows + 1) * sizeof(decltype(*row_starts)) +
           sizeof(GhostEntryCoordinateFormat) * ghost_entries.capacity();
  }

  std::size_t memory_consumption_sell_c() const
  {
    int total_size = 0;
    for (unsigned int i = 0; i < n_rows / chunk_size; ++i){
      total_size += cl[i] * chunk_size;
    }
    return total_size * sizeof(unsigned int) + total_size * sizeof(Number) + 
          (n_rows / chunk_size) * sizeof(unsigned int) + (n_rows / chunk_size) * sizeof(unsigned int);
  }

  void print_CRS() {
    // for check convert CRS to SELL-C
    std::cout << "First value CRS: " << values[0] << std::endl;
    for (int i=0; i<32; ++i){
      std::cout << std::setw(2) << i << ": ";

      for (int j=row_starts[i]; j<row_starts[i+1]; j++){
        std::cout << std::setw(8) << std::setprecision(4) << values[j] << ",";
      }
      std::cout << std::endl;
    }
  }

  void print_SELLC(){
    std::vector<unsigned int> row_lengths(n_rows);
    for (unsigned int i = 0; i < n_rows; ++i)
      row_lengths[i] = row_starts[i + 1] - row_starts[i];
    
    // calculate max row length for each chunk
    std::vector<unsigned int> max_row_length_chunk(n_rows / chunk_size);
    // go through each chunk
    for (unsigned int i = 0; i < n_rows / chunk_size; ++i) {
      max_row_length_chunk[i] = 0;
      // go through each row in chunk
      for (unsigned int j = 0; j < chunk_size; ++j) {
        if (row_lengths[i * chunk_size + j] > max_row_length_chunk[i]) {
          max_row_length_chunk[i] = row_lengths[i * chunk_size + j];
        }
      }
    }
    std::cout << "First value SELL-C: " << values_sell_c[0] << std::endl;

    for (int i=0; i<chunk_size; ++i){
      std::cout << std::setw(2) << i << ": ";
      for (int j=0; j<max_row_length_chunk[0]; ++j){
        std::cout << std::setw(8) << std::setprecision(4) << values_sell_c[j*chunk_size + i] << ",";
      }
      std::cout << std::endl;
    }
  }

  void convert_to_sell_c(){
    // calculate each row length
    std::vector<unsigned int> row_lengths(n_rows);
    for (unsigned int i = 0; i < n_rows; ++i)
      row_lengths[i] = row_starts[i + 1] - row_starts[i];
    
    // calculate max row length for each chunk
    std::vector<unsigned int> max_row_length_chunk(n_rows / chunk_size);
    // go through each chunk
    for (unsigned int i = 0; i < n_rows / chunk_size; ++i) {
      max_row_length_chunk[i] = 0;
      // go through each row in chunk
      for (unsigned int j = 0; j < chunk_size; ++j) {
        if (row_lengths[i * chunk_size + j] > max_row_length_chunk[i]) {
          max_row_length_chunk[i] = row_lengths[i * chunk_size + j];
        }
      }
    }

    // convert CRS to SELL-C
    int chunk_offset = 0;
    int offset = 0;
    // go through each chunk
    for (int k = 0; k < n_rows / chunk_size; k++){
      chunk_offset = max_row_length_chunk[k] * chunk_size;
      // go through each row in chunk
      for (int i = 0; i < chunk_size; i++){
        // go through each elements in a row
        for (int j = 0; j < max_row_length_chunk[k]; j++){
          if (j < row_lengths[k * chunk_size + i]){
            column_indices_sell_c[offset + j * chunk_size + i] = column_indices[row_starts[k * chunk_size + i] + j];
            values_sell_c[offset + j * chunk_size + i] = values[row_starts[k * chunk_size + i] + j];
          }
          else{
            column_indices_sell_c[offset + j * chunk_size + i] = 0;
            values_sell_c[offset + j * chunk_size + i] = 0;
          }
        }
      }
      offset += chunk_offset;
    }
    // set the starting offset for each chunk
    cs[0] = 0;
    for (int i = 1; i < n_rows / chunk_size; i++){
      cs[i] = cs[i-1] + max_row_length_chunk[i-1] * chunk_size;
    }

    // set the length of each chunk, equals to the max row length in each chunk
    for (int i = 0; i < n_rows / chunk_size; i++){
      cl[i] = max_row_length_chunk[i];
    }
  }

private:
  MPI_Comm      communicator;
  std::size_t   n_rows;
  std::size_t * row_starts;
  unsigned int *column_indices;
  Number *      values;
  std::size_t   n_global_nonzero_entries;
  MemorySpace   memory_space;

  struct GhostEntryCoordinateFormat
  {
    unsigned int index_within_result;
    unsigned int index_within_offproc_vector;
    Number       value;
  };
  std::vector<GhostEntryCoordinateFormat> ghost_entries;

  std::vector<std::pair<unsigned int, std::vector<unsigned int>>> send_indices;
  mutable std::vector<Number>                                     send_data;
  std::vector<std::pair<unsigned int, unsigned int>> receive_indices;
  mutable std::vector<Number>                        receive_data;

  // for sell-c structure
  Number *      values_sell_c;
  unsigned int *column_indices_sell_c;
  unsigned int *cl;
  unsigned int *cs;
  unsigned int chunk_size = 32;

  Number *      values_sellc_cuda;
  unsigned int *column_indices_sellc_cuda;
  unsigned int *cl_cuda;
  unsigned int *cs_cuda;
};


#endif
