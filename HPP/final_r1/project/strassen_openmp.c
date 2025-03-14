#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

void allocate_mem(int*** arr, int n);
void free_mem(int** arr, int n);
int rand_int(int N);
int ** naive_mul(int n, int **a, int **b);
bool check(int n, int **mat1, int **mat2);
void print_matrix(int N, int **a);
int ** sum(int n, int **a, int **b, bool sum);
int ** divide_matrix(int n, int **a, int i, int j);
int ** strassen(int n, int **a, int **b);
int ** combine_matrix(int n, int **a, int **b, int **c, int **d);
int check_matrix(int N);
int next_power_of_2(int N);


int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <number of threads>\n", argv[0]);
    return -1;
  }

  int i, j, n, N;
  int **a;
  int **b;
  int **c;
  int **d;
  int Nmax = 10; // random numbers in [0, N]

  n = atoi(argv[1]); // get matrix size from command line
  N = atoi(argv[2]);

  // check the matrix size
  if (check_matrix(n) == -1) {
      n = next_power_of_2(n);
  }

  // create matrix A
  allocate_mem(&a, n);
  for ( i = 0 ; i < n ; i++ )
    for ( j = 0 ; j < n ; j++ )
      a[i][j] = rand_int(Nmax);

  // print matrix A
  // printf("Matrix A is:\n");
  // print_matrix(n, a);

  // create matrix B
  allocate_mem(&b, n);
  for ( i = 0 ; i < n ; i++ )
    for ( j = 0 ; j < n ; j++ )
      b[i][j] = rand_int(Nmax);
  
  double start_strassen = omp_get_wtime();
  
  #pragma omp parallel num_threads(N)
  {
    #pragma omp single
    {
      c = strassen(n, a, b);
    }
  }

  double end_strassen = omp_get_wtime();
  d = naive_mul(n, a, b);

  printf("Parallel Strassen Runtime: %lf\n", end_strassen - start_strassen);

  // check the accuracy of the Strassen algorithm
  bool res = check(n, c, d);
  if (res == true) {
      printf("The dimension of the matrix is %d, and the result is right!\n", n);
  }

  free_mem(a, n);
  free_mem(b, n);
  free_mem(c, n);
  free_mem(d, n);
  return 0;
}



// allocate memory for a 2D array
void allocate_mem(int*** arr, int n)
{
  int i;
  *arr = (int**)malloc(n*sizeof(int*));
  for(i=0; i<n; i++)
    (*arr)[i] = (int*)malloc(n*sizeof(int));
}

// free memory for a 2D array
void free_mem(int** arr, int n)
{
  int i;
  for(i=0; i<n; i++)
    free(arr[i]);
  free(arr);
}

// generate a random interger in [0, N - 1]
int rand_int(int N)
{
  int val = -1;
  while( val < 0 || val >= N )
    {
      val = (int)(N * (double)rand()/RAND_MAX);
    }
  return val;
}

// naive matrix multiplication function, jik has the best loop order
int ** naive_mul(int n, int **a, int **b)
{
  int **c;
  allocate_mem(&c, n);
  int i, j, k;

  #pragma omp parallel for collapse(2) 
    for (j = 0; j < n; j++) {
      for (i = 0; i < n; i++) {
        int sum = 0;
        for (k = 0; k < n; k++) {
          sum += a[i][k] * b[k][j];
        }
        c[i][j] = sum;
      }
    }

  return c;
}

// check the accuracy of the Strassen algorithm
bool check(int n, int **mat1, int **mat2) 
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mat1[i][j] != mat2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

// strassen function
int ** strassen(int n, int **a, int **b)
{

  if (n <= 32) {
    return naive_mul(n, a, b);
  }

  else {
    int new_size = n / 2;
    // create the submatrix
    int **a11, **a12, **a21, **a22;
    int **b11, **b12, **b21, **b22;
    int **c11, **c12, **c21, **c22;
    int **m1, **m2, **m3, **m4, **m5, **m6, **m7;

    
    // divide the matrix into 4 sub-matrixs
    a11 = divide_matrix(n, a, 0, 0);
    a12 = divide_matrix(n, a, 0, new_size);
    a21 = divide_matrix(n, a, new_size, 0);
    a22 = divide_matrix(n, a, new_size, new_size);

    b11 = divide_matrix(n, b, 0, 0);
    b12 = divide_matrix(n, b, 0, new_size);
    b21 = divide_matrix(n, b, new_size, 0);
    b22 = divide_matrix(n, b, new_size, new_size);

    #pragma omp task shared(m7)
    {
      int **minus_a = sum(new_size, a12, a22, false);
      int **add_b = sum(new_size, b21, b22, true);
      m7 = strassen(new_size, minus_a, add_b);
      free_mem(minus_a, new_size);
      free_mem(add_b, new_size);
    }

    #pragma omp task shared(m6)
    {
      int **minus_a1 = sum(new_size, a21, a11, false);
      int **add_b1 = sum(new_size, b11, b12, true);
      m6 = strassen(new_size, minus_a1, add_b1);
      free_mem(minus_a1, new_size);
      free_mem(add_b1, new_size);
    }

    #pragma omp task shared(m5)
    {
      int **add_a = sum(new_size, a11, a12, true);
      m5 = strassen(new_size, add_a, b22);
      free_mem(add_a, new_size);
    }

    #pragma omp task shared(m4)
    {
      int **minus_b = sum(new_size, b21, b11, false);
      m4 = strassen(new_size, a22, minus_b);
      free_mem(minus_b, new_size);
    }

    #pragma omp task shared(m3)
    {
      int **minus_b1 = sum(new_size, b12, b22, false);
      m3 = strassen(new_size, a11, minus_b1);
      free_mem(minus_b1, new_size);
    }

    #pragma omp task shared(m2)
    {
      int **add_a1 = sum(new_size, a21, a22, true);
      m2 = strassen(new_size, add_a1, b11);
      free_mem(add_a1, new_size);
    }

    #pragma omp task shared(m1)
    {
      int **add_a2 = sum(new_size, a11, a22, true);
      int **add_b2 = sum(new_size, b11, b22, true);
      m1 = strassen(new_size, add_a2, add_b2);
      free_mem(add_a2, new_size);
      free_mem(add_b2, new_size);
    }

    #pragma omp taskwait

    free_mem(a11, new_size);
    free_mem(a12, new_size);
    free_mem(a21, new_size);
    free_mem(a22, new_size);
    free_mem(b11, new_size);
    free_mem(b12, new_size);
    free_mem(b21, new_size);
    free_mem(b22, new_size);

    #pragma omp task shared(c11)
    {
      int **add_m17 = sum(new_size, m1, m7, true);
      int **sub_m45 = sum(new_size, m4, m5, false);
      c11 = sum(new_size, add_m17, sub_m45, true);
      free_mem(add_m17, new_size);
      free_mem(sub_m45, new_size);
    }

    #pragma omp task shared(c12)
    {
      c12 = sum(new_size, m3, m5, true);
    }
    
    #pragma omp task shared(c21)
    {
      c21 = sum(new_size, m2, m4, true);
    }

    #pragma omp task shared(c22)
    {
      int **sub_m12 = sum(new_size, m1, m2, false);
      int **sum_m36 = sum(new_size, m3, m6, true);
      c22 = sum(new_size, sub_m12, sum_m36, true);
      free_mem(sub_m12, new_size);
      free_mem(sum_m36, new_size);
    }

    #pragma omp taskwait

    free_mem(m1, new_size);
    free_mem(m2, new_size);
    free_mem(m3, new_size);
    free_mem(m4, new_size);
    free_mem(m5, new_size);
    free_mem(m6, new_size);
    free_mem(m7, new_size);

    // combine the submatrix
    int **c;
    c = combine_matrix(new_size, c11, c12, c21, c22);

    free_mem(c11, new_size);
    free_mem(c12, new_size);
    free_mem(c21, new_size);
    free_mem(c22, new_size);

    return c;
  }
}

void print_matrix(int N, int **a) 
{
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
}


int ** sum(int n, int **a, int **b, bool sum)
{
  int i, j;
  int **result;
  allocate_mem(&result, n);
  // #pragma omp parallel for collapse(2)
    for (i=0; i<n; i++) {
      for (j=0; j<n; j++) {
        if (sum) {
          result[i][j] = a[i][j] + b[i][j];
        }
        else {
          result[i][j] = a[i][j] - b[i][j];
        }
      }
    }
  return result;
}

// divide the matrix into 4 sub-matrixs
int ** divide_matrix(int n, int **a, int i, int j) 
{
    int **result;
    allocate_mem(&result, n / 2);
    int x, y;
    for (x = 0; x < n / 2; x++) {
        for (y = 0; y < n / 2; y++) {
            result[x][y] = a[x + i][y + j];
        }
    }
    return result;
}

int ** combine_matrix(int n, int **a, int **b, int **c, int **d)
{
  int size = n * 2;
  int **result;
  allocate_mem(&result, size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i < n && j < n) {
        result[i][j] = a[i][j];
      }
      else if (i < n) {
        result[i][j] = b[i][j-n];
      }
      else if (j < n) {
        result[i][j] = c[i-n][j];
      }
      else {
        result[i][j] = d[i-n][j-n];
      }
    }
  }
  return result;
}

int check_matrix(int N) 
{
    if (N <= 0) {
        printf("Matrix size must be larger than 0!");
        return 0;
    }
    // check if the matrix size is a power of 2
    while( N != 1) {
        if (N % 2 != 0) {
            printf("Matrix size must be a power of 2!\n");
            return -1;
        }
        N = N / 2;
    }
    return 1;
}

// find the next power of 2
int next_power_of_2(int N) 
{
    int i = 1;
    while (i < N) {
        i <<= 1;
    }
    return i;
}