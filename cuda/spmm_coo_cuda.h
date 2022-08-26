#pragma once

#include <torch/extension.h>

#define CHECK_INPUT_DIM(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_coo_cuda(
    const torch::Tensor row,
    const torch::Tensor col,
    const torch::optional<torch::Tensor> optional_value,
    torch::Tensor mat,
    int64_t dim_size,
    std::string reduce);