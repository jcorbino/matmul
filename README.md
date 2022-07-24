# matmul
Sandbox for matrix multiplication, just having fun with fundamental algorithms

* For the CPU version (.cpp), you need to have CBLAS installed

* The GPU version (.cu), uses device shared memory to ensure data reuse, the matrix multiplication is done using tiling approach. Of course the best option is to call `cublasDgemm(...)` instead of reinventing the wheel
