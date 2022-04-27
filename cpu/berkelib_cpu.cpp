#include "berkelib_cpu.h"
#include "reducer.h"

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

std::tuple<torch::Tensor, torch::Tensor> max_mul_cpu_forward(
    const torch::Tensor src,
    const torch::Tensor edge_start,
    const torch::Tensor edge_end,
    int64_t res_dim)
{
    // check input before prossesing
    CHECK_CPU(src);
    CHECK_CPU(edge_start);
    CHECK_CPU(edge_end);
    CHECK_INPUT(edge_start.size(0) == edge_end.size(0));

    auto hidden_dim = size(src, 1);
    auto edge_count = edge_end.numel();

    // create out and arg_out Tensor with given out_dim
    auto res_dims = src.sizes().vec();
    res_dims[0] = res_dim;
    torch::Tensor res = torch::empty(res_dims, src.options());
    torch::Tensor arg_out = torch::full_like(res, src.size(0), edge_start.options());

    // max sparse matrix multiplication
    AT_DISPATCH_FLOATING_TYPES(src.type(), "_", [&] {
        res.fill_(std::numeric_limits<scalar_t>::lowest());
        auto src_data = src.data_ptr<scalar_t>();
        auto res_data = res.data_ptr<scalar_t>();
        auto arg_out_data = arg_out.data_ptr<int64_t>();
        auto edge_start_data = edge_start.data_ptr<int64_t>();
        auto edge_end_data = edge_end.data_ptr<int64_t>();

        for (auto e = 0; e < edge_count; e++)
        {
            for (auto h = 0; h < hidden_dim; h++)
            {
               Reducer<scalar_t, MAX>::update(
                res_data + edge_end_data[e]*hidden_dim + h, 
                src_data[edge_start_data[e]*hidden_dim + h],
                arg_out_data + edge_end_data[e]*hidden_dim + h, edge_start_data[e]); 
            }
        }
        
        res.masked_fill_(res == std::numeric_limits<scalar_t>::lowest(), (scalar_t)0); 
    });

    return std::make_tuple(res, arg_out);
}