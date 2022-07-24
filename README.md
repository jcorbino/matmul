# matmul
Sandbox for matrix multiplication, just having fun with fundamental algorithms.

* For the CPU version (.cpp), you need to have CBLAS installed. IJK, IJK (with transposed B), and IKJ approaches are implemented

* The GPU version (.cu), uses shared memory to ensure data reuse, and the tiling approach. Of course the best option is to call `cublasDgemm(...)` instead of reinventing the wheel

**NOTE:** In both implementations I use one-dimensional dynamically allocated arrays.
