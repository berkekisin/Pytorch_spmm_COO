#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#include "berkelib_cuda.h"
#include "reducer.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS


inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

template <typename scalar_t>
__global__ void max_mul_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ edge_start, 
    const int64_t* __restrict__ edge_end,
    scalar_t* __restrict__ res,
    size_t hidden_dim,
    size_t N)
{
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;      

        Reducer<scalar_t, MAX>::atomic_write(
           res + edge_end[edge_index]*hidden_dim + hidden_dim_index, 
            src[ edge_start[edge_index]*hidden_dim + hidden_dim_index]);  
    }
}

template <typename scalar_t>
__global__ void max_mul_arg_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ edge_start, 
    const int64_t* __restrict__ edge_end,
    scalar_t* __restrict__ res,
    int64_t* __restrict__ arg_out,
    size_t hidden_dim,
    size_t N)
{
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id < N){
        int edge_index = thread_id / hidden_dim;
        int hidden_dim_index = thread_id % hidden_dim;

        if(res[edge_end[edge_index]*hidden_dim + hidden_dim_index] == src[edge_start[edge_index]*hidden_dim + hidden_dim_index]){
            arg_out[edge_end[edge_index]*hidden_dim + hidden_dim_index] = edge_start[edge_index];
        }
    }
}

std::tuple<torch::Tensor,torch::Tensor> max_mul_cuda_forward(
    const torch::Tensor src, 
    const torch::Tensor edge_start, 
    const torch::Tensor edge_end,
    int64_t res_dim)
{
    //check input
    CHECK_INPUT(src);
    CHECK_INPUT_DIM(edge_start.size(0) == edge_end.size(0));
    CHECK_INPUT(edge_start);
    CHECK_INPUT(edge_end);

    size_t hidden_dim = size(src, 1);
    size_t N = edge_end.numel()*hidden_dim;

    //create out and arg_out Tensor with given out_dim
    auto res_dims = src.sizes().vec();
    res_dims[0] = res_dim;
    torch::Tensor res = torch::empty(res_dims, src.options());
    torch::Tensor arg_out = torch::full_like(res,src.size(0),edge_start.options());
    
    AT_DISPATCH_FLOATING_TYPES(src.type(), "_", [&] {
        res.fill_(std::numeric_limits<scalar_t>::lowest());
        auto src_data = src.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        auto arg_out_data = arg_out.data_ptr<int64_t>();
        auto edge_start_data = edge_start.data_ptr<int64_t>();
        auto edge_end_data = edge_end.data_ptr<int64_t>();

        max_mul_kernel<scalar_t><<<BLOCKS(N), THREADS>>>(
            src_data,
            edge_start_data,
            edge_end_data,
            res_data,
            hidden_dim,
            N);

        res.masked_fill_(res == std::numeric_limits<scalar_t>::lowest(), (scalar_t)0);

        max_mul_arg_kernel<scalar_t><<<BLOCKS(N), THREADS>>>(
            src_data,
            edge_start_data,
            edge_end_data,
            res_data,
            arg_out_data,
            hidden_dim,
            N);   
    });

    checkCuda(cudaGetLastError());
    
    return std::make_tuple(res,arg_out);   
}