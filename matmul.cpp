// g++ -Wall -O3 -fopenmp matmul.cpp -o matmul -lblas
// ./matmul <ROWS_A> <COLS_B> <COLS_A>

#include <iostream>
#include <random>
#include <chrono>
#include <cblas.h>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "usage: ./matmul <ROWS_A> <COLS_B> <COLS_A>\n";
        return 1;
    }

    // Output formatting
    cout.precision(4);
    cout.setf(ios::fixed);

    // Random numbers
    mt19937_64 rnd(random_device{}());
    uniform_real_distribution<double> dist(0, 1);

    // Dimensions
    uint m = atoi(argv[1]); // Rows of A, C
    uint n = atoi(argv[2]); // Cols of B, C
    uint o = atoi(argv[3]); // Cols of A, Rows of B

    // If you know the dimensions at compile time:
    // double (*A)[COLS] = malloc(sizeof *A * ROWS);
    // Use A[i][j]...
    // But in this case we don't, and VLAs are not part of the C++ standard
    double *A = new double[m * o];
    double *B = new double[o * n];
    double *BT = new double[n * o];
    double *C1 = new double[m * n]();
    double *C2 = new double[m * n]();
    double *C3 = new double[m * n]();
    double *C4 = new double[m * n]();

    // Fill A with random numbers
    for (uint k = 0; k < m * o; ++k) // Manually collapsed loop
        A[k] = dist(rnd);

    // Fill B (and B^T) with random numbers
    for (uint k = 0; k < o * n; ++k) // Manually collapsed loop
    {
        uint i = k / n, j = k % n;
        B[k] = BT[j * o + i] = dist(rnd);
    }

#ifdef DEBUG
    // Print A
    cout << "A:\n";
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < o; ++j)
            cout << A[i * o + j] << ' ';
        cout << '\n';
    }
    cout << '\n';

    // Print B
    cout << "B:\n";
    for (uint i = 0; i < o; ++i)
    {
        for (uint j = 0; j < n; ++j)
            cout << B[i * n + j] << ' ';
        cout << '\n';
    }
    cout << '\n';

    // Print B^T
    cout << "B^T:\n";
    for (uint i = 0; i < n; ++i)
    {
        for (uint j = 0; j < o; ++j)
            cout << BT[i * o + j] << ' ';
        cout << '\n';
    }
    cout << '\n';
#endif

    auto start = chrono::steady_clock::now();
    // C-Wrapper of BLAS for C := alpha*op(A)*op(B) + beta*C
    //                                                             alpha   lda   ldb beta
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, o, 1.0, A, o, B, n, 0.0, C1, n);
    auto stop = chrono::steady_clock::now();
    cout << "Time with BLAS: " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";

    start = chrono::steady_clock::now();
#pragma omp parallel for
    for (uint i = 0; i < m; ++i)
        for (uint j = 0; j < n; ++j)
            for (uint k = 0; k < o; ++k)
                C2[i * n + j] += A[i * o + k] * B[k * n + j];
    stop = chrono::steady_clock::now();
    cout << "Time with ijk: " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";

    start = chrono::steady_clock::now();
#pragma omp parallel for
    for (uint i = 0; i < m; ++i)
        for (uint j = 0; j < n; ++j)
        {
            double sum = 0.0;
            for (uint k = 0; k < o; ++k)
                sum += A[i * o + k] * BT[j * o + k];
            C3[i * n + j] = sum;
        }
    stop = chrono::steady_clock::now();
    cout << "Time with ijk (B^T): " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";

    start = chrono::steady_clock::now();
#pragma omp parallel for
    for (uint i = 0; i < m; ++i)
        for (uint k = 0; k < o; ++k)
            for (uint j = 0; j < n; ++j)
                C4[i * n + j] += A[i * o + k] * B[k * n + j];
    stop = chrono::steady_clock::now();
    cout << "Time with ikj: " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";

#ifdef DEBUG
    // Print C1 (BLAS)
    cout << "\nC1:\n";
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < n; ++j)
            cout << C1[i * n + j] << ' ';
        cout << '\n';
    }
    cout << '\n';

    // Print C2 (ijk)
    cout << "C2:\n";
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < n; ++j)
            cout << C2[i * n + j] << ' ';
        cout << '\n';
    }
    cout << '\n';

    // Print C3 (ijk B^T)
    cout << "C3:\n";
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < n; ++j)
            cout << C3[i * n + j] << ' ';
        cout << '\n';
    }
    cout << '\n';

    // Print C4 (ikj)
    cout << "C4:\n";
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < n; ++j)
            cout << C4[i * n + j] << ' ';
        cout << '\n';
    }
#endif

    // Clean up
    delete[] A;
    delete[] B;
    delete[] BT;
    delete[] C1;
    delete[] C2;
    delete[] C3;
    delete[] C4;

    return 0;
}
