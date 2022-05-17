#include <torch/script.h>

#include "utils.h"
//#include "cpu/berkelib_cpu.h"

//#ifdef WITH_CUDA
#include "cuda/scatter_edge_cuda.h"
//#endif

std::tuple<torch::Tensor, torch::Tensor> scatter_edge_forward(
    const torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim,
    std::string reduce)
{
  if (src.device().is_cuda())
  {
//#ifdef WITH_CUDA
    return scatter_edge_cuda(src, edge_start, edge_end, res_dim, reduce);
//#else
   //AT_ERROR("Not compiled with CUDA support");
//#endif
  }
  //else
  //{
  // return max_mul_cpu_forward(src, edge_start, edge_end, res_dim);
  //}
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
    // ctx->saved_data["dim"] = dim_size;
    ctx->saved_data["src_shape"] = src.sizes();
    // auto result = max_mul_cuda_forward(src, edge_start, edge_end, dim_size);
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
    // auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[0] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(0, arg_out, grad_out, "add");
    grad_in = grad_in.narrow(0, 0, src_shape[0] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("scatter_edge_max", &scatter_edge_max, "Max Sparse Mul forward");
}
