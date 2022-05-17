import torch
from scatter_edge import scatter_edge_max
from torch.nn import Linear
from torch_scatter import scatter_max, scatter_sum

device = 'cuda:0'

def test(x, edge_start, edge_end,maxmul=True):
  
    if maxmul:
        x,y = scatter_edge_max(x,edge_start, edge_end, x.shape[0])
    else:
        x,y = scatter_max(x[edge_start], edge_end,dim=0, dim_size = x.shape[0])
    
    return (x,y)

if __name__ == "__main__":
    src = torch.rand((10,10),dtype=torch.float32, device=device, requires_grad = True)
    edge_start = torch.randint(10,(50,),device=device)
    edge_end = torch.randint(10,(50,),device=device)
    k = scatter_edge_max(src,edge_start, edge_end, src.shape[0])[0]
    l = scatter_max(src[edge_start], edge_end,dim=0 ,dim_size = src.shape[0])[0]
    print(torch.equal(k,l))

    #torch edge scatter
    res = test(src,edge_start,edge_end)
    arg = res[1]
    res = res[0].sum()
    res.backward()
    a = src.grad
    
    #torch scatter
    src.grad= None
    res2 = test(src,edge_start,edge_end,False)
    arg2 = res2[1]
    res2 = res2[0].sum()
    res2.backward()
    b = src.grad
    print(torch.equal(a,b))
   


