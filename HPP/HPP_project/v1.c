#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
 
int rand_int(int N)
{
  int val = -1;
  while( val < 0 || val >= N )
    {
      val = (int)(N * (double)rand()/RAND_MAX);
    }
  return val;
}

void print_matrix(int N, int **a) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
}

bool check_accuracy(int n, int **mat1, int **mat2) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mat1[i][j] != mat2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void allocate_mem(int*** arr, int n)
{
  int i;
  *arr = (int**)malloc(n*sizeof(int*));
  for(i=0; i<n; i++)
    (*arr)[i] = (int*)malloc(n*sizeof(int));
}

void free_mem(int** arr, int n)
{
  int i;
  for(i=0; i<n; i++)
    free(arr[i]);
  free(arr);
}

void sum(int n, int **a, int **b, int **c)
{
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
}

void sub(int n, int **a, int **b, int **c)
{
  int i, j;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
}

// divide the matrix into 4 sub-matrixs
int ** divide_matrix(int n, int **a, int i, int j) {
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

// naive matrix multiplication function
void naive_mul(int n, int **a, int **b, int **c)
{
  int i, j, k;
  for (k=0; k<n; k++) {
    for (i=0; i<n; i++) {
      int x = a[i][k];
      for (j=0; j<n; j++)
	c[i][j] += x * b[k][j];   
    }
  }
}

// strassen function
void strassen(int n, int **a, int **b, int **c)
{
  printf("matrix a is:\n");
  print_matrix(n, a);
  if (n <= 2) {
    printf("hello\n");
    naive_mul(n, a, b, c);
    return;
  }

  else {
    int i, j, k;
    int new_size = n / 2;
    // create the submatrix
    int **a11, **a12, **a21, **a22;
    int **b11, **b12, **b21, **b22;
    int **c11, **c12, **c21, **c22;
    int **m1, **m2, **m3, **m4, **m5, **m6, **m7;

    int **temp1, **temp2;

    // divide the matrix into 4 sub-matrixs
    a11 = divide_matrix(n, a, 0, 0);
    a12 = divide_matrix(n, a, 0, new_size);
    a21 = divide_matrix(n, a, new_size, 0);
    a22 = divide_matrix(n, a, new_size, new_size);

    b11 = divide_matrix(n, b, 0, 0);
    b12 = divide_matrix(n, b, 0, new_size);
    b21 = divide_matrix(n, b, new_size, 0);
    b22 = divide_matrix(n, b, new_size, new_size);

    // allocate the memory for the submatrix
    allocate_mem(&c11, new_size);
    allocate_mem(&c12, new_size);
    allocate_mem(&c21, new_size);
    allocate_mem(&c22, new_size);

    allocate_mem(&m1, new_size);
    allocate_mem(&m2, new_size);
    allocate_mem(&m3, new_size);
    allocate_mem(&m4, new_size);
    allocate_mem(&m5, new_size);
    allocate_mem(&m6, new_size);
    allocate_mem(&m7, new_size);
    
    allocate_mem(&temp1, new_size);
    allocate_mem(&temp2, new_size);

    // calculate the m1 to m7
    sum(new_size, a11, a22, temp1);
    printf("a11:\n");
    print_matrix(new_size, a11);
    printf("a22:\n");
    print_matrix(new_size, a22);
    printf("temp1:\n");
    printf("new_size is %d\n", new_size);
    print_matrix(new_size, temp1);
    sum(new_size, b11, b22, temp2);
    strassen(new_size, temp1, temp2, m1);

    sum(new_size, a21, a22, temp1);
    strassen(new_size, temp1, b11, m2);

    sub(new_size, b12, b22, temp2);
    strassen(new_size, a11, temp2, m3);

    sub(new_size, b21, b11, temp2);
    strassen(new_size, a22, temp2, m4);

    sum(new_size, a11, a12, temp1);
    strassen(new_size, temp1, b22, m5);

    sub(new_size, a21, a11, temp1);
    sum(new_size, b11, b12, temp2);
    strassen(new_size, temp1, temp2, m6);

    sub(new_size, a12, a22, temp1);
    sum(new_size, b21, b22, temp2);
    strassen(new_size, temp1, temp2, m7);

    // calculate the c11 to c22
    sum(new_size, m3, m5, c12);
    sum(new_size, m2, m4, c21);
    
    sum(new_size, m1, m4, temp1);
    sub(new_size, temp1, m5, temp2);
    sum(new_size, temp2, m7, c11);

    sub(new_size, m1, m2, temp1);
    sum(new_size, temp1, m3, temp2);
    sum(new_size, temp2, m6, c22);

    // combine the submatrix
    for (i = 0; i < new_size; i++) {
      for (j = 0; j < new_size; j++) {
          c[i][j] = c11[i][j];
          c[i][j + new_size] = c12[i][j];
          c[i + new_size][j] = c21[i][j];
          c[i + new_size][j + new_size] = c22[i][j];
      }
    }

    // free the memory
    free_mem(a11, new_size);
    free_mem(a12, new_size);
    free_mem(a21, new_size);
    free_mem(a22, new_size);

    free_mem(b11, new_size);
    free_mem(b12, new_size);
    free_mem(b21, new_size);
    free_mem(b22, new_size);

    free_mem(c11, new_size);
    free_mem(c12, new_size);
    free_mem(c21, new_size);
    free_mem(c22, new_size);

    free_mem(m1, new_size);
    free_mem(m2, new_size);
    free_mem(m3, new_size);
    free_mem(m4, new_size);
    free_mem(m5, new_size);
    free_mem(m6, new_size);
    free_mem(m7, new_size);

    free_mem(temp1, new_size);
    free_mem(temp2, new_size);
  }
}


int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <N>\n", argv[0]);
    return -1;
  }

  int i, j, n;
  int **a;
  int **b;
  int **c;
  int **d;
  int Nmax = 10; // random numbers in [0, N]

  n = atoi(argv[1]);

  // create matrix A
  allocate_mem(&a, n);
  for ( i = 0 ; i < n ; i++ )
    for ( j = 0 ; j < n ; j++ )
      a[i][j] = rand_int(Nmax);

  // print matrix A
  printf("Matrix A is:\n");
  print_matrix(n, a);

  // create matrix B
  allocate_mem(&b, n);
  for ( i = 0 ; i < n ; i++ )
    for ( j = 0 ; j < n ; j++ )
      b[i][j] = rand_int(Nmax);

  // print matrix B
  // printf("Matrix B is:\n");
  // print_matrix(n, b);

  // create a matrix to store the result
  allocate_mem(&c, n);
  allocate_mem(&d, n);

  strassen(n, a, b, c);
  naive_mul(n, a, b, d);

  print_matrix(n, c);
  printf("\n");
  print_matrix(n, d);
  // check the accuracy of the Strassen algorithm
  bool check = check_accuracy(n, c, d);
  if (check == true) {
      printf("The dimension of the matrix is %d, and the result is right!\n", n);
  }

  free_mem(a, n);
  free_mem(b, n);
  free_mem(c, n);
  free_mem(d, n);
  return 0;
}
