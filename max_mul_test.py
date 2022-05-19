import torch
from torch_scatter import scatter_max, scatter_sum, scatter_mean
from scatter_edge import scatter_edge_max, scatter_edge_sum, scatter_edge_mean
#import berkelib
from torch.nn import Linear

import time

device = 'cuda:0'
hidden_dim = 10
lin1 = Linear(hidden_dim,hidden_dim, bias=False)
lin2 = Linear(hidden_dim,hidden_dim, bias=False)
lin1.to(device)
lin2.to(device)
def test(x, edge_start, edge_end,maxmul=True):
    hidden_dim = x.shape[1]
    #x_out = x.shape[0] - 1
    #x = lin1(x)
    if maxmul:
        pass
        x = scatter_edge_mean(x,edge_start, edge_end, x.shape[0])
    else:
        x = scatter_mean(x[edge_start], edge_end,dim=0, dim_size = x.shape[0])
    #x = lin2(x)
    
    return (x,0)

if __name__ == "__main__":
    src = torch.rand((100,20),dtype=torch.float64, device=device, requires_grad = True)
    edge_start = torch.randint(100,(100,),device=device)
    edge_end = torch.randint(100,(100,),device=device)
    #src = torch.tensor([[10,2,3],[2,8,5]],dtype=torch.float32, device=device, requires_grad = True)
    #edge_start = torch.tensor([0,1,1],device=device)
    #edge_end = torch.tensor([0,0,1],device=device)
    #res = scatter_edge_mean(src,edge_start, edge_end, 2).sum()
   
    b = scatter_sum(src[edge_start],edge_end,dim = 0,dim_size=src.shape[0])
    
    res = test(src,edge_start,edge_end)
    arg = res[1]
    res = res[0].sum()
    res.backward()
    a = src.grad
    src.grad= None

    res2 = test(src,edge_start,edge_end,False)
    arg2 = res2[1]
    res2 = res2[0].sum()
    #print(res2)
    res2.backward()
    b = src.grad
    #print(arg)
    #print(arg2)
    #print(a)
    #print(b)
    print(a-b)
    #print(torch.equal(torch.Tensor(list(src.grad.shape)),torch.Tensor(list(src.shape))))


