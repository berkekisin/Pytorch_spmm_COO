#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> max_mul_cpu_forward(
    const torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim);

