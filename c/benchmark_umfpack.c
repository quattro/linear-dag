#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <umfpack.h>
#include <cholmod.h>

// Function to generate random double between 0 and 1
double rand_double() {
    return (double)rand() / RAND_MAX;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_file = argv[1];
    cholmod_common c;
    cholmod_start(&c);  // Initialize CHOLMOD

    // Load matrix from .mtx file
    FILE *f = fopen(matrix_file, "r");
    if (!f) {
        perror("Error opening file");
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }

    cholmod_sparse *A = cholmod_read_sparse(f, &c);
    fclose(f);

    if (!A) {
        fprintf(stderr, "Error reading sparse matrix from file.\n");
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }

    // Matrix dimensions
    int n = A->nrow;
    int m = A->ncol;
    if (n != m) {
        fprintf(stderr, "Error: Matrix must be square to solve (I - A)x = b.\n");
        cholmod_free_sparse(&A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }
    printf("Matrix dimensions: %d x %d\n", n, m);

    
    cholmod_sparse *I = cholmod_speye(n, n, CHOLMOD_REAL, &c);  // n x n identity matrix

    // Modify A to become (I - A)
    double alpha[2] = {1.0, 0.0};   // Scaling factor for I (real part 1.0, imaginary part 0.0)
    double beta[2] = {-1.0, 0.0};   // Scaling factor for A (real part -1.0, imaginary part 0.0)
    int mode = CHOLMOD_REAL;        // Indicating the matrix type (real-valued)

    // Perform C = I - A
    cholmod_sparse *I_minus_A = cholmod_add(I, A, alpha, beta, mode, 1, &c);
    cholmod_free_sparse(&I, &c);

    int *Ap = (int *)I_minus_A->p;
    int *Ai = (int *)I_minus_A->i;
    double *Ax = (double *)I_minus_A->x;

    // Generate random dense m x 10 matrix b
    int num_cols_b = 100;
    double *b = malloc(m * num_cols_b * sizeof(double));
    if (!b) {
        fprintf(stderr, "Memory allocation failed for b.\n");
        cholmod_free_sparse(&A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }
    srand((unsigned int)time(NULL));  // Seed for randomness
    for (int i = 0; i < m * num_cols_b; i++) {
        b[i] = rand_double();  // Random value between 0 and 1
    }

    // Solution array x
    double *x = malloc(m * num_cols_b * sizeof(double));
    if (!x) {
        fprintf(stderr, "Memory allocation failed for x.\n");
        free(b);
        cholmod_free_sparse(&A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }

    // Start timing the UMFPACK solve
    clock_t start = clock();

    void *Symbolic, *Numeric;
    int status;
    status = umfpack_di_symbolic(n, m, Ap, Ai, Ax, &Symbolic, NULL, NULL);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "UMFPACK symbolic factorization failed.\n");
        free(x);
        free(b);
        cholmod_free_sparse(&A, &c);
        cholmod_free_sparse(&I_minus_A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }

    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
        fprintf(stderr, "UMFPACK numeric factorization failed.\n");
        free(x);
        free(b);
        cholmod_free_sparse(&A, &c);
        cholmod_free_sparse(&I_minus_A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }

    // Solve (I - A)x = b for each column in b
    for (int j = 0; j < num_cols_b; j++) {
        status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax,
                                  &x[j * m], &b[j * m], Numeric, NULL, NULL);
        if (status != UMFPACK_OK) {
            fprintf(stderr, "UMFPACK solve failed for column %d.\n", j);
            umfpack_di_free_numeric(&Numeric);
            free(x);
            free(b);
            cholmod_free_sparse(&A, &c);
            cholmod_free_sparse(&I_minus_A, &c);
            cholmod_finish(&c);
            return EXIT_FAILURE;
        }
    }
    umfpack_di_free_numeric(&Numeric);

    // End timing
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time to solve (I - A)x=b for m x 10 matrix b: %f seconds\n", elapsed_time);

    // Clean up
    free(x);
    free(b);
    cholmod_free_sparse(&A, &c);
    cholmod_free_sparse(&I_minus_A, &c);
    cholmod_finish(&c);

    return EXIT_SUCCESS;
}