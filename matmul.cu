// nvcc --compiler-options -Wall matmul.cu -o matmul
// ./matmul <ROWS_A> <COLS_B> <COLS_A>

#include <iostream>
#include <random>
#include <chrono>

#define TILE_WIDTH 16 // 16x16 = 256 CUDA threads per block

using namespace std;

// Macro to check for errors
#define checkCudaErrors(value)    \
    {                             \
        check((value), __LINE__); \
    }
inline void check(cudaError_t code, int line)
{
    if (code != cudaSuccess)
    {
        cerr << cudaGetErrorString(code) << ", at line: " << line << endl;
        exit(code);
    }
}

// Kernel
__global__ void matmul(const double *A, const double *B, double *C, uint hA, uint wA, uint wB)
{
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

    uint tx = threadIdx.x;
    uint ty = threadIdx.y;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;

    uint row = by * TILE_WIDTH + ty;
    uint col = bx * TILE_WIDTH + tx;

    double value = 0.0;

    for (uint ph = 0; ph < (wA + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
    {
        if (row < hA && ph * TILE_WIDTH + tx < wA)
            As[ty][tx] = A[row * wA + ph * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;

        if (col < wB && ph * TILE_WIDTH + ty < wA)
            Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * wB + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (uint k = 0; k < TILE_WIDTH; ++k)
            value += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < hA && col < wB)
        C[row * wB + col] = value;
}

int main(int argc, char *argv[])
{
    // Host and device pointers
    double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    uint mem_size_A, mem_size_B, mem_size_C;

    if (argc != 4)
    {
        cerr << "usage: ./matmul <ROWS_A> <COLS_B> <COLS_A>\n";
        return 1;
    }

    // Random numbers
    mt19937_64 rnd(random_device{}());
    uniform_real_distribution<double> dist(0, 1);

    // Dimensions
    uint m = atoi(argv[1]); // Rows of A, C
    uint n = atoi(argv[2]); // Cols of B, C
    uint o = atoi(argv[3]); // Cols of A, Rows of B

    mem_size_A = sizeof(double) * m * o;
    mem_size_B = sizeof(double) * o * n;
    mem_size_C = sizeof(double) * m * n;

    // Allocate memory on the host
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    // Allocate memory on the device (GPU)
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // Fill matrix A with random numbers
    for (uint k = 0; k < m * o; ++k)
        h_A[k] = dist(rnd);

    // Fill matrix B with random numbers
    for (uint k = 0; k < o * n; ++k)
        h_B[k] = dist(rnd);

    auto start = chrono::steady_clock::now();

    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

    matmul<<<grid, threads>>>(d_A, d_B, d_C, m, o, n);
    checkCudaErrors(cudaPeekAtLastError());

    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    auto stop = chrono::steady_clock::now();
    cout << "Elapsed time (including data transfers): " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";

// Check result!
#ifdef DEBUG
    double *tmp = (double *)calloc(m * n, sizeof(double));
    for (uint i = 0; i < m; ++i)
        for (uint k = 0; k < o; ++k)
            for (uint j = 0; j < n; ++j)
                tmp[i * n + j] += h_A[i * o + k] * h_B[k * n + j];

    for (uint i = 0; i < m; ++i)
        for (uint j = 0; j < n; ++j)
            if (fabs(h_C[i * n + j] - tmp[i * n + j]) > 0.01)
                cout << "Matrices differ!\n";

    free(tmp);
#endif

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}
