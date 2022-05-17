# Pytorch_scatter_edge

Current implementation includes scatter operation with max reduction and only supports cuda.

Consider the following szenario: \n
-A source tensor with size: n*k \n
-Tensor edge start which contains start nodes of edges with size: l \n
-Tensor edge end which containt end nodes of edges with size: l \n 

We have l*k threads in total. There is one group of k threads for each edge and they are responsible for one row of result tensor. For example for an edge [2,6] that starts from node 2 and ends in node 6, the group of k threads is responsible for row 6 and updates it with values from row 2 of source tensor. One row has length k so each thread is responsible for one indix of the row.
Below is an illustration of out thread hierarchie:

![alt text](https://github.com/berkekisin/Pytorch_scatter_edge/blob/main/thread.jpg?raw=true)
