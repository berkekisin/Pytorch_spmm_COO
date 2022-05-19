# Pytorch_scatter_edge

The current implementation includes scatter operation with max, sum and mean reductions and only supports Cuda.
<br />
Consider the following scenario: <br />
-A source tensor with size: n.k <br />
-Tensor edge start which contains start nodes of edges with size: l <br />
-Tensor edge end which contains end nodes of edges with size: l <br />
<br />
There are l.k threads in total. There is a group of k threads for each edge and each group is responsible for one row of result tensor. For example for an edge [2,6] that starts from node 2 and ends in node 6, the group of k threads is responsible for row 6 and updates it with values from row 2 of the source tensor. One row has length k so each thread is responsible for one index of the row.
Below is an illustration of thread hierarchy:

![alt text](https://github.com/berkekisin/Pytorch_scatter_edge/blob/main/thread.jpg?raw=true)
