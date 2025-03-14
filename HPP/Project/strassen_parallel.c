# include <stdio.h>
# include <stdlib.h>

int check_matrix(int N) {
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
int next_power_of_2(int N) {
    int i = 1;
    while (i < N) {
        i = i * 2;
    }
    return i;
}

// allocate memory for a matrix
int ** create_matrix(int n) {
    int *data = (int *)malloc(n * n * sizeof(int));
    int **array = (int **)malloc(n * sizeof(int *));
    // check if the memory is allocated
    if (data == NULL || array == NULL) {
        printf("Matrix memory allocation failed!");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        array[i] = &(data[i * n]);
    }
    return array;
}

// randomly generate a matrix
void random_matrix(int n, int N, int **A) {
    int i, j;
    printf("%d %d\n", n, N);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 10;  // generate a random number between 0 and 9
        }
    }
    
    if (n != N) {
        for (i = N; i < n; i++) {
            for (j = N; j < n; j++) {
                A[i][j] = 0;
            }
        }
    }
}

// function to print the matrix
void print_matrix(int N, int **A) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }
}

int ** add_matrix(int n, int **A, int **B) {
    int i, j;
    int **result;
    result = create_matrix(n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

int ** sub_matrix(int n, int **A, int **B) {
    int i, j;
    int **result;
    result = create_matrix(n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// divide the matrix into 4 sub-matrixs
int ** divide_matrix(int n, int **A, int i, int j) {
    int **result;
    result = create_matrix(n / 2);
    int x, y;
    for (x = 0; x < n / 2; x++) {
        for (y = 0; y < n / 2; y++) {
            result[x][y] = A[x + i][y + j];
        }
    }
    return result;
}

int ** strassen(int n, int **A, int **B) {
    int **C;
    C = create_matrix(n);
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return C;
    }

    // decline the matrix demension
    int m = n / 2;
    // printf("m = %d\n", m);
    int **A11 = divide_matrix(n, A, 0, 0);
    int **A12 = divide_matrix(n, A, 0, m);
    int **A21 = divide_matrix(n, A, m, 0);
    int **A22 = divide_matrix(n, A, m, m);
    int **B11 = divide_matrix(n, B, 0, 0);
    int **B12 = divide_matrix(n, B, 0, m);
    int **B21 = divide_matrix(n, B, m, 0);
    int **B22 = divide_matrix(n, B, m, m);

    // calculate the 7 sub-matrixs
    int **M1 = strassen(m, add_matrix(m, A11, A22), add_matrix(m, B11, B22));
    int **M2 = strassen(m, add_matrix(m, A21, A22), B11);
    int **M3 = strassen(m, A11, sub_matrix(m, B12, B22));
    int **M4 = strassen(m, A22, sub_matrix(m, B21, B11));
    int **M5 = strassen(m, add_matrix(m, A11, A12), B22);
    int **M6 = strassen(m, sub_matrix(m, A21, A11), add_matrix(m, B11, B12));
    int **M7 = strassen(m, sub_matrix(m, A12, A22), add_matrix(m, B21, B22));

    // calculate the 4 sub-matrixs of the result matrix
    int **C11 = add_matrix(m, sub_matrix(m, add_matrix(m, M1, M4), M5), M7);
    int **C12 = add_matrix(m, M3, M5);
    int **C21 = add_matrix(m, M2, M4);
    int **C22 = add_matrix(m, add_matrix(m, sub_matrix(m, M1, M2), M3), M6);

    // combine the 4 sub-matrixs into a matrix
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            C[i][j] = C11[i][j];
            C[i][j + m] = C12[i][j];
            C[i + m][j] = C21[i][j];
            C[i + m][j + m] = C22[i][j];
        }
    }

    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
    free(C11);
    free(C12);
    free(C21);
    free(C22);
    
    return C;
}

int main(int argc, char *argv[]) {
    // check the number of arguments
    if (argc != 2) {
        printf("Usage: %s <matrix size>", argv[0]);
        return -1;
    }

    // create the matrixs to be multiplied
    int N = atoi(argv[1]);
    int n = N;

    // check the matrix size
    if (check_matrix(N) == -1) {
        n = next_power_of_2(N);
    }

    int **A = create_matrix(n); // should return a array of pointers 
    random_matrix(n, N, A);
    printf("Matrix A:\n");
    print_matrix(n, A);
    printf("\n");

    int **B = create_matrix(n);
    random_matrix(n, N, B);
    print_matrix(n, B);
    printf("\n");
    
    int **C = strassen(n, A, B);
    print_matrix(N, C);

    free(A);
    free(B);
    free(C);
    return 0;
}