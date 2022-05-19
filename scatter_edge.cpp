#include <torch/script.h>

#include "utils.h"
#include "cuda/scatter_edge_cuda.h"

std::tuple<torch::Tensor, torch::Tensor> scatter_edge_forward(
    const torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce)
{
  if (src.device().is_cuda())
    return scatter_edge_cuda(src, edge_start, edge_end, res_dim, reduce);
  else
    AT_ERROR("Source Tensor not in GPU!");
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ScatterEdgeMax : public torch::autograd::Function<ScatterEdgeMax>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = scatter_edge_forward(src, edge_start, edge_end, res_dim, "max");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    ctx->save_for_backward({arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0];
    auto arg_out = ctx->get_saved_variables()[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out, "add");
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class ScatterEdgeSum : public torch::autograd::Function<ScatterEdgeSum>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    // ctx->saved_data["dim"] = dim_size;
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = scatter_edge_forward(src, edge_start, edge_end, res_dim, "sum");
    auto out = std::get<0>(result);
    ctx->save_for_backward({edge_start, edge_end});
    ctx->mark_non_differentiable({edge_start, edge_end});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0];
    auto edge_start = ctx->get_saved_variables()[0];
    auto edge_end = ctx->get_saved_variables()[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto result = scatter_edge_forward(grad_out, edge_end, edge_start, src_shape[0], "sum");
    auto grad_in = std::get<0>(result);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};

class ScatterEdgeMean : public torch::autograd::Function<ScatterEdgeMean>
{
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable edge_start,
                               Variable edge_end,
                               int64_t res_dim)
  {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = scatter_edge_forward(src, edge_start, edge_end, res_dim, "sum");
    auto out = std::get<0>(result);
    // compute degree of elements in result tensor
    auto ones = torch::ones(res_dim, src.options());
    result = scatter_edge_forward(ones, edge_start, edge_end, res_dim, "sum");
    auto degree = std::get<0>(result);
    degree.masked_fill_(degree < 1, 1);
    // divide result tensor by degree
    degree = broadcast(degree, out, 0);
    if (out.is_floating_point())
      out.true_divide_(degree);
    else
      out.div_(degree, "floor");
    ctx->save_for_backward({edge_start, edge_end, degree});
    ctx->mark_non_differentiable({edge_start, edge_end, degree});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
  {
    auto grad_out = grad_outs[0].clone();
    auto saved = ctx->get_saved_variables();
    auto edge_start = saved[0];
    auto edge_end = saved[1];
    auto degree = saved[2];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    grad_out.true_divide_(degree);
    auto result = scatter_edge_forward(grad_out, edge_end, edge_start, src_shape[0], "sum");
    auto grad_in = std::get<0>(result);
    return {grad_in, Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor> scatter_edge_max(const torch::Tensor src,
                                                          const torch::Tensor edge_start,
                                                          const torch::Tensor edge_end,
                                                          int64_t res_dim)
{
  auto result = ScatterEdgeMax::apply(src, edge_start, edge_end, res_dim);

  return std::make_tuple(result[0], result[1]);
}

torch::Tensor scatter_edge_sum(const torch::Tensor src,
                               const torch::Tensor edge_start,
                               const torch::Tensor edge_end,
                               int64_t res_dim)
{

  return ScatterEdgeSum::apply(src, edge_start, edge_end, res_dim)[0];
}

torch::Tensor scatter_edge_mean(const torch::Tensor src,
                                const torch::Tensor edge_start,
                                const torch::Tensor edge_end,
                                int64_t res_dim)
{

  return ScatterEdgeMean::apply(src, edge_start, edge_end, res_dim)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("scatter_edge_sum", &scatter_edge_sum, "Sum Sparse Mul forward");
  m.def("scatter_edge_max", &scatter_edge_max, "Max Sparse Mul forward");
  m.def("scatter_edge_mean", &scatter_edge_mean, "Mean Sparse Mul forward");
}