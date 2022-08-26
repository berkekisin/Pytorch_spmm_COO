#include <torch/script.h>
#include <vector>

#include "cuda/spmm_coo_cuda.h"
#include "utils.h"


std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_coo_fw(
    const torch::Tensor row,
    const torch::Tensor col,
    const torch::optional<torch::Tensor> optional_value,
    const torch::Tensor mat,
    int64_t dim_size,
    std::string reduce)
{
    if (row.device().is_cuda())
    {
        return spmm_coo_cuda(row, col, optional_value, mat, dim_size, reduce);
    }
    else
    {
        AT_ERROR("Row Tensor not in GPU!");
        //return spmm_coo_cpu(row, col, optional_value, mat, dim_size, reduce);
    }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SPMMSum : public torch::autograd::Function<SPMMSum>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable value,
                                 Variable mat,
                                 int64_t dim_size,
                                 bool has_value)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        ctx->saved_data["has_value"] = has_value;
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;

        auto out = std::get<0>(spmm_coo_fw(row, col, opt_value, mat, dim_size, "sum"));
        ctx->save_for_backward({row, col, value});
        ctx->mark_non_differentiable({row, col, value});

        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto has_value = ctx->saved_data["has_value"].toBool();
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto row = saved[0];
        auto col = saved[1];
        auto value = saved[2];

        // compute gradient of mat
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;
        auto grad_mat = std::get<0>(spmm_coo_fw(col, row, opt_value, grad_out, mat_shape[0], "sum"));

        return {Variable(), Variable(), Variable(), grad_mat, Variable(), Variable()};
    }
};

class SPMMMean : public torch::autograd::Function<SPMMMean>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable value,
                                 Variable mat,
                                 int64_t dim_size,
                                 bool has_value)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        ctx->saved_data["has_value"] = has_value;
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;
        auto out = std::get<0>(spmm_coo_fw(row, col, opt_value, mat, dim_size, "sum"));

        // compute degree of elements in result tensor
        auto ones = torch::ones(size(mat, 0), mat.options());
        auto degree = std::get<0>(spmm_coo_fw(row, col, torch::nullopt, ones, dim_size, "sum"));
        degree.masked_fill_(degree < 1, 1);

        // divide result tensor by degree
        degree = broadcast(degree, out, 0);
        if (out.is_floating_point())
            out.true_divide_(degree);
        else
            out.div_(degree, "floor");
        ctx->save_for_backward({row, col, degree, value});
        ctx->mark_non_differentiable({row, col, degree, value});

        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto has_value = ctx->saved_data["has_value"].toBool();
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        auto grad_out = grad_outs[0].clone();
        auto saved = ctx->get_saved_variables();
        auto row = saved[0];
        auto col = saved[1];
        auto degree = saved[2];
        auto value = saved[3];

        grad_out.true_divide_(degree);
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;
        auto grad_mat = std::get<0>(spmm_coo_fw(col, row, opt_value, grad_out, mat_shape[0], "sum"));

        return {Variable(), Variable(), Variable(), grad_mat, Variable(), Variable()};
    }
};

class SPMMMax : public torch::autograd::Function<SPMMMax>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable value,
                                 Variable mat,
                                 int64_t dim_size,
                                 bool has_value)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        ctx->saved_data["has_value"] = has_value;
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;

        auto result = spmm_coo_fw(row, col, opt_value, mat, dim_size, "max");
        auto out = std::get<0>(result);
        auto arg_out = std::get<1>(result).value();
        ctx->save_for_backward({arg_out, value,row});
        ctx->mark_non_differentiable({row, col, arg_out, value});

        return {out, arg_out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto has_value = ctx->saved_data["has_value"].toBool();
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto arg_out = saved[0];
        auto value = saved[1];
        auto row = saved[2];
        auto invalid_arg_mask = arg_out == row.size(0);
        arg_out = arg_out.masked_fill(invalid_arg_mask, 0);
        grad_out = grad_out.masked_fill(invalid_arg_mask, 0);

        //compute gradient of mat
        auto grad_mat = torch::zeros(mat_shape, grad_out.options());
        if (has_value){
            grad_out = value.index_select(0, arg_out.flatten()).view_as(grad_out).mul_(grad_out);
            arg_out = row.index_select(0,arg_out.flatten()).view_as(grad_out);
        }
        grad_mat.scatter_(0, arg_out, grad_out,"add");

        return {Variable(), Variable(),Variable(), grad_mat, Variable(),Variable()};
    }
};


class SPMMMin : public torch::autograd::Function<SPMMMin>
{
public:
    static variable_list forward(AutogradContext *ctx, Variable row,
                                 Variable col,
                                 Variable value,
                                 Variable mat,
                                 int64_t dim_size,
                                 bool has_value)
    {
        ctx->saved_data["mat_shape"] = mat.sizes();
        ctx->saved_data["has_value"] = has_value;
        torch::optional<torch::Tensor> opt_value = torch::nullopt;
        if (has_value)
            opt_value = value;

        auto result = spmm_coo_fw(row, col, opt_value, mat, dim_size, "min");
        auto out = std::get<0>(result);
        auto arg_out = std::get<1>(result).value();
        ctx->save_for_backward({arg_out, value,row});
        ctx->mark_non_differentiable({row, col, arg_out, value});

        return {out, arg_out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs)
    {
        auto has_value = ctx->saved_data["has_value"].toBool();
        auto mat_shape = list2vec(ctx->saved_data["mat_shape"].toIntList());
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto arg_out = saved[0];
        auto value = saved[1];
        auto row = saved[2];
        auto invalid_arg_mask = arg_out == row.size(0);
        arg_out = arg_out.masked_fill(invalid_arg_mask, 0);
        grad_out = grad_out.masked_fill(invalid_arg_mask, 0);

        //compute gradient of mat
        auto grad_mat = torch::zeros(mat_shape, grad_out.options());
        if (has_value){
            grad_out = value.index_select(0, arg_out.flatten()).view_as(grad_out).mul_(grad_out);
            arg_out = row.index_select(0,arg_out.flatten()).view_as(grad_out);
        }
        grad_mat.scatter_(0, arg_out, grad_out,"add");

        return {Variable(), Variable(),Variable(), grad_mat, Variable(),Variable()};
    }
};

torch::Tensor spmm_coo_sum(const torch::Tensor row,
                                      const torch::Tensor col,
                                      const torch::optional<torch::Tensor> opt_value,
                                      const torch::Tensor mat,
                                      int64_t dim_size)
{
    auto value = opt_value.has_value() ? opt_value.value() : col;
    return SPMMSum::apply(row, col, value, mat, dim_size, opt_value.has_value())[0];
}

torch::Tensor spmm_coo_mean(const torch::Tensor row,
                                       const torch::Tensor col,
                                       const torch::optional<torch::Tensor> opt_value,
                                       const torch::Tensor mat,
                                       int64_t dim_size)
{
    auto value = opt_value.has_value() ? opt_value.value() : col;
    return SPMMMean::apply(row, col, value, mat, dim_size, opt_value.has_value())[0];
}

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_max(const torch::Tensor row,
                                                                 const torch::Tensor col,
                                                                 const torch::optional<torch::Tensor> opt_value,
                                                                 const torch::Tensor mat,
                                                                 int64_t dim_size)
{
    auto value = opt_value.has_value() ? opt_value.value() : col;
    auto result = SPMMMax::apply(row, col, value, mat, dim_size, opt_value.has_value());

    return std::make_tuple(result[0], result[1]);
}

std::tuple<torch::Tensor, torch::Tensor> spmm_coo_min(const torch::Tensor row,
                                                                 const torch::Tensor col,
                                                                 const torch::optional<torch::Tensor> opt_value,
                                                                 const torch::Tensor mat,
                                                                 int64_t dim_size)
{
    auto value = opt_value.has_value() ? opt_value.value() : col;
    auto result = SPMMMin::apply(row, col, value, mat, dim_size, opt_value.has_value());

    return std::make_tuple(result[0], result[1]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("spmm_coo_sum", &spmm_coo_sum, "Sum Sparse coo Mul forward");
    m.def("spmm_coo_mean", &spmm_coo_mean, "Mean Sparse coo Mul forward");
    m.def("spmm_coo_max", &spmm_coo_max, "Max Sparse coo Mul forward");
    m.def("spmm_coo_min", &spmm_coo_min, "Min Sparse coo Mul forward");
}

//static auto registry = torch::RegisterOperators()
//                           .op("torch_sparse::spmm_coo_sum", &spmm_coo_sum)
//                           .op("torch_sparse::spmm_coo_mean", &spmm_coo_mean)
//                           .op("torch_sparse::spmm_coo_min", &spmm_coo_min)
//                           .op("torch_sparse::spmm_coo_max", &spmm_coo_max);