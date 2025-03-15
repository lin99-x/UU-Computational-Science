#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double find_median (double *data, int len);
double find_mean (double *data, int len);
int compare (const void *a, const void *b);


int main(int argc, char *argv)
{
    // check if the number of arguments is correct
    if (argc != 4)
    {
        printf("Usage: %s <input file> <output file> <pivot strategies>\n", argv[0]);
        return -1;
    }

    char *inputfile = argv[1];
    char *outputfile = argv[2];

    FILE *fp;

    int rank, size, N;
    double *total_array, *local;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // rank 0 read input file
    if (rank == 0) {
        fp = fopen(inputfile, "r");
        if (fp == NULL) {
            perror("Error opening input file");
            return -1;
        }

        // read the number of elements to sort
        fscanf(fp, "%d", &N);

        total_array = (double *)malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            fscanf(fp, "%d", &total_array[i]);
        }
    }

    // broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = N / size;  /* should be divisible */

    local = (double *)malloc(local_size * sizeof(double));

    // start timer
    double start_time = MPI_Wtime();

    // scatter the array to all processes
    MPI_Scatter(total_array, local_size, MPI_DOUBLE, local, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // sort the local array
    qsort(local, local_size, sizeof(double), compare);

    // perform global sort
    
}



/* define the compare function */
int compare (const void *a, const void *b)
{
    // the sort is in ascending order
    if (*(double *a) > *(double *b)) 
        return 1;
    else if (*(double *a) == *(double *b))
        return 0;
    else
        return -1;
}

/* define pivot strategies */ 
double find_median (double *data, int len)
{
    // is there a possibility that length equals to 0?

    if (len == 1) {
        return data[0];
    }
    // if length is odd then return the middle element
    if (len % 2 == 1) {
        return data[len / 2];
    }
    // if length is even then return the average of the two middle elements
    return (data[len / 2 - 1] + data[len / 2]) / 2;
}

double find_mean (double *data, int len)
{
    double sum = 0, mean = 0;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    mean = sum / (double)len;
    return mean;
}