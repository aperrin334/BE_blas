#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cblas.h>

// Preprocessor defined global variables (must recompile program to change values)
#define Nstart 128 // number of rows and columns of base matrix A, B, C
#define step 64   // step to grow the problem size at each iteration
#define nstep 32    // number of steps to perform
#define ncase 10   // number of cases to run to obtain a mean time value
#define BLOCK 64   // Block size for blocking version

// Declaration of functions (defined after the main program to improve readability)
void init_matrix(double*, int, int, int, double);
double norm_matrix(double*, int, int, int);
void print_matrix(double*, int, int, int);
void inhouse_add(double*, double*, double*, int, int);
void inhouse_add_reorder(double*, double*, double*, int, int);
void inhouse_dot(double*, double*, double*, int, int);
void inhouse_blocking(double*, double*, double*, int, int);
void infos_openmp(void);

//-----------------------------------------------------------------------------
//  Main of the program returning an integer (0 if correct execution, other else)
//-----------------------------------------------------------------------------

int main(void)
{
  // Declaration of local variables
  int N;              // number of rows and columns of matrix A, B and C
  int ld;             // step in memory for the A matrix to reach the next column
  double *A;          // matrix A defined as a single pointer
  double *B;          // matrix B defined as a single pointer
  double *C;          // matrix C defined as a single pointer
  const char *which_method; // To store method used and print it to screen

  // Declaration of the file to output results
  FILE *timings = fopen("timings.csv", "w");
  if (timings == NULL) {
    fprintf(stderr, "Error: Cannot open timings.csv for writing\n");
    return 1;
  }

  fprintf(timings, "#dimension\t\ttime(s)\t\tGflops/s\n");

  // Useful variables of performance estimation
  double time, flops;

  // OpenMP informations
  infos_openmp();

  // Set the dimensions according to the base matrices dimensions predefined
  N  = Nstart;
  ld = N + 1;  // Padding to avoid cache line conflicts

  // Compute C = A + B or C = A x B
  // Use -DUSE_ADD, -DUSE_BLAS{1,2,3} or -DUSE_MUL
  // at compilation according to each method

  for (int k = 0; k < nstep; k++)
  {
    // Allocate matrices in memory
    A = (double*)malloc(N * ld * sizeof(double));
    B = (double*)malloc(N * ld * sizeof(double));
    C = (double*)malloc(N * ld * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "Error: Memory allocation failed for N=%d\n", N);
      free(A); free(B); free(C);
      fclose(timings);
      return 1;
    }

    // Initialize matrices entries
    init_matrix(A, ld, N, N, 1.0);
    init_matrix(B, ld, N, N, 2.0);

    // First terminal screen output 
    printf("Problem dimensions: N=%d\n", N);

    // To calculate time spent (robust OpenMP routine)
    time = omp_get_wtime();

    #if defined USE_ADD

      which_method = "Inhouse Add";

      for (int c = 0; c < ncase; c++)
      {
        // Initialization of the C matrix
        init_matrix(C, ld, N, N, 0.0);
        // Call to the routine coded by our hands defined below
        inhouse_add(A, B, C, N, ld);
        //inhouse_add_reorder(A, B, C, N, ld);
      }

    #elif defined USE_BLAS1

      // BLAS1
      // Compute each entry via a scalar product between the row of A and the column of B
      // Two nested do loops to cover all entries of C

      which_method = "BLAS1";

      for (int c = 0; c < ncase; c++)
      {
        // Initialization of the C matrix
        init_matrix(C, ld, N, N, 0.0);
        // Loop over the columns
        for (int j = 0; j < N; j++)
          // Loop over the rows
          for (int i = 0; i < N; i++)
            C[i + ld*j] = cblas_ddot(N, &A[i], ld, &B[ld*j], 1);
      }

    #elif defined USE_BLAS2

      // BLAS2
      // Compute the entire column entry of the matrix C
      // via the matrix/vector product of the matrix A with an entire column entry of the matrix B

      which_method = "BLAS2";

      for (int c = 0; c < ncase; c++)
      {
        // Initialization of the C matrix
        init_matrix(C, ld, N, N, 0.0);
        // Call the external linked BLAS2 routine
        for (int j = 0; j < N; j++)
          cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1., A, ld, &B[ld*j], 1, 0., &C[ld*j], 1);
      }

    #elif defined USE_BLAS3

      // BLAS3
      // The matrix/matrix product is done in one time calling the BLAS dgemm routine

      which_method = "BLAS3";

      for (int c = 0; c < ncase; c++)
      {
        // Initialization of the C matrix
        init_matrix(C, ld, N, N, 0.0);
        // Call the external linked BLAS3 routine
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1., A, ld, B, ld, 0., C, ld);
      }

    #elif defined USE_MUL

      // Compute the matrix/matrix product by our hands

      which_method = "Inhouse Multiplication";

      for (int c = 0; c < ncase; c++)
      {
        // Initialization of the C matrix
        init_matrix(C, ld, N, N, 0.0);
        // Call to the routine coded by our hands defined below
        // inhouse_dot(A, B, C, N, ld);
        inhouse_blocking(A, B, C, N, ld);
      }

    #else

      printf("\nNo -DUSE_xxx specified at compilation line (call your professor :D)\n");
      free(A); free(B); free(C);
      fclose(timings);
      return 1;

    #endif

    // Final calculation of mean time spent according to ncase loop 
    time = (omp_get_wtime() - time) / (double)ncase;
    
    // Computing the number of arithmetic operations
    #if defined USE_ADD
    flops = (double)N * (double)N;
    #else
    flops = 2. * (double)N * (double)N * (double)N;
    #endif

    // Informations output in terminal screen
    printf("Method used     = %s\n", which_method);
    printf("Frobenius Norm  = %f\n", norm_matrix(C, N, N, ld));
    printf("Mean total time = %f s\n", time);
    printf("Gflops/s        = %f\n\n", flops / (time * 1e9));
    
    // Performances written in the 'timings.txt' file
    fprintf(timings, "%d\t\t%f\t\t%f\n", N, time, flops / (time * 1e9));
  
    // Free memory
    free(A);
    free(B);
    free(C);

    // Change dimensions of the problem
    N  = N + step;
    ld = N + 1;
  }

  fclose(timings);

  // Final return with 0 value as the program is finished with correct execution
  return 0;
}

//-----------------------------------------------------------------------------
//  Compute the C = A + B by our hands
//-----------------------------------------------------------------------------

void inhouse_add(double* A, double* B, double* C, int N, int ld)
{
  #pragma omp // TO BE FINISHED
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      C[i + ld*j] = A[i + ld*j] + B[i + ld*j];
}

// Something to change to enhance speed computation rather than inhouse_add
void inhouse_add_reorder(double* A, double* B, double* C, int N, int ld)
{
  #pragma omp // TO BE FINISHED
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      C[i + ld*j] = A[i + ld*j] + B[i + ld*j];
}

//-----------------------------------------------------------------------------
//  Compute the C = A x B by our hands
//-----------------------------------------------------------------------------

void inhouse_dot(double* A, double* B, double* C, int N, int ld)
{
  #pragma omp parallel for shared(A, B, C) schedule(runtime)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < N; k++)
      for (int i = 0; i < N; i++)
        C[i + ld*j] += A[i + ld*k] * B[k + ld*j];
}

void inhouse_blocking(double* A, double* B, double* C, int N, int ld)
{
  #pragma omp  parallel for schedule(runtime)
  for (int j = 0; j < N; j += BLOCK)
    for (int k = 0; k < N; k += BLOCK)
      for (int i = 0; i < N; i += BLOCK)
        for (int jj = 0; jj < BLOCK; jj++)
          for (int kk = 0; kk < BLOCK; kk++)
            for (int ii = 0; ii < BLOCK; ii++)
              C[(i+ii) + ld*(j+jj)] += A[(i+ii) + ld*(k+kk)] * B[(k+kk) + ld*(j+jj)]; // TO BE FINISHED (indices!)
}

//-----------------------------------------------------------------------------
//  Initialization of a Matrix A(nrow,ncol)
//-----------------------------------------------------------------------------

void init_matrix(double *A, int ld, int nrow, int ncol, double cst)
{
  #pragma omp parallel for schedule(runtime)
  for (int j = 0; j < ncol; j++)
    for (int i = 0; i < nrow; i++)
      A[i + ld*j] = cst / sqrt((double)nrow) / sqrt((double)ncol);
}

//-----------------------------------------------------------------------------
//  Compute the Frobenius norm of a Matrix A(nrow,ncol)
//-----------------------------------------------------------------------------

double norm_matrix(double* A, int nrow, int ncol, int ld)
{
  double norm = 0.;
  #pragma omp parallel for schedule(runtime)
  for (int j = 0; j < ncol; j++)
    for (int i = 0; i < nrow; i++)
      norm += A[i + ld*j] * A[i + ld*j];
  return sqrt(norm);
}

//-----------------------------------------------------------------------------
//  Output in terminal screen the Matrix A(nrow,ncol)
//-----------------------------------------------------------------------------

void print_matrix(double* A, int nrow, int ncol, int ld)
{
  for (int i = 0; i < nrow; i++) {
    printf("(");
    for (int j = 0; j < ncol; j++)
      printf("%5.2f ", A[i + j*ld]);
    printf(")\n");
  }
  printf("\n");
}

//-----------------------------------------------------------------------------
//  To output OpenMP informations
//-----------------------------------------------------------------------------

void infos_openmp(void)
{
  omp_sched_t kind;
  int chunk_size;

  printf("\nParallel execution with a maximum of %d threads callable\n\n", omp_get_max_threads());

  omp_get_schedule(&kind, &chunk_size);

  if (kind == omp_sched_static)
    printf("Scheduling static  with chunk = %d\n\n", chunk_size);
  else if (kind == omp_sched_dynamic)
    printf("Scheduling dynamic with chunk = %d\n\n", chunk_size);
  else if (kind == omp_sched_auto)
    printf("Scheduling auto    with chunk = %d\n\n", chunk_size);
  else if (kind == omp_sched_guided)
    printf("Scheduling guided  with chunk = %d\n\n", chunk_size);
}
