import torch
from scatter_edge import scatter_edge_max
#import berkelib
from torch.nn import Linear
from torch_scatter import scatter_max, scatter_sum
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
        x,y = scatter_edge_max(x,edge_start, edge_end, x.shape[0])
    else:
        x,y = scatter_max(x[edge_start], edge_end,dim=0, dim_size = x.shape[0])
    #x = lin2(x)
    
    return (x,y)

if __name__ == "__main__":
    src = torch.rand((10,10),dtype=torch.float32, device=device, requires_grad = True)
    edge_start = torch.randint(10,(2,),device=device)
    edge_end = torch.randint(10,(2,),device=device)
    a = scatter_edge_max(src,edge_start, edge_end, src.shape[0])[0]
    b = scatter_max(src[edge_start],edge_end,dim = 0,dim_size=src.shape[0])[0]
   
    
    res = test(src,edge_start,edge_end)
    arg = res[1]
    res = res[0].sum()
    #print(res)
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
    print(torch.equal(a,b))
    #print(torch.equal(torch.Tensor(list(src.grad.shape)),torch.Tensor(list(src.shape))))


