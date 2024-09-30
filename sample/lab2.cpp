#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    if (argc != 3) {
        if (rank == 0) { // Only process 0 prints the error message
            fprintf(stderr, "must provide exactly 2 arguments!\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels_local = 0;
    unsigned long long pixels_total = 0;

    // Divide the range of x across processes
    unsigned long long chunk_size = r / size;
    unsigned long long start_x = rank * chunk_size;
    unsigned long long end_x = (rank == size - 1) ? r : start_x + chunk_size;

    // Each process computes its own portion of pixels
    for (unsigned long long x = start_x; x < end_x; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        pixels_local += y;
        pixels_local %= k;
    }

    // Reduce all local results to get the total sum at process 0
    MPI_Reduce(&pixels_local, &pixels_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 prints the final result
    if (rank == 0) {
        printf("%llu\n", (4 * pixels_total) % k);
    }

    MPI_Finalize();  // Clean up the MPI environment
    return 0;
}
