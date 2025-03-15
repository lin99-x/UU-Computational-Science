#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "prop.h"

#define bins 20 // number of bins
#define T 100 // time horizon

int main(int argc, char* argv[]) {

    // should have one argument specifying the number of local experiments
    // shoule have one argument specifying the output file name
    if (argc != 3) {
        printf("Usage: %s <number of local experiments> <output file name>\n", argv[0]);
        return -1;
    }

    // initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // get the number of local experiments
    int local_experiments = atoi(argv[1]);
    int N = local_experiments * size;

    // get the output file name
    char *output_file = argv[2];

    // initialize the state vector x
    int x0[7] = {900, 900, 30, 330, 50, 270, 20};

    // initialize the vector w
    double *w = (double *)calloc(15, sizeof(double));

    // local final time matrix
    int *local_final_matrix = (int *)calloc(7 * local_experiments, sizeof(int));

    // initialize the matrix P
    int *P = (int *)calloc(15 * 7, sizeof(int));
    P[0*7+0] = 1; 
    P[1*7+0] = -1; 
    P[2*7+0] = -1; P[2*7+2] = 1;
    P[3*7+1] = 1; 
    P[4*7+1] = -1; 
    P[5*7+1] = -1; P[5*7+3] = 1;
    P[6*7+2] = -1; 
    P[7*7+2] = -1; P[7*7+4] = 1; 
    P[8*7+3] = -1;
    P[9*7+3] = -1; P[9*7+5] = 1; 
    P[10*7+4] = -1; 
    P[11*7+4] = -1; P[11*7+6] = 1; 
    P[12*7+5] = -1; 
    P[13*7+0] = 1; P[13*7+6] = -1;
    P[14*7+6] = -1;

    // start timer
    double start_time = MPI_Wtime();
    double one_quarter_time = 0;
    double half_time = 0;
    double three_quarter_time = 0;
    double final_time = 0;

    // for each process, they will execute the local experiments and form the final vector x at time T
    for (int i=0; i<local_experiments; i++) {
        // set a final simulation time T, current time t, initial state x=x0
        double t = 0;
        int x[7];
        for (int j=0; j<7; j++) {
            x[j] = x0[j];
        }

        // simulate the system until time T
        while (t < T) {
            // compute propensities
            prop(x, w);

            // compute a0
            double a0 = 0;
            for (int j=0; j<15; j++) {
                a0 += w[j];
            }

            // generate two random numbers
            double u1 = (double)rand() / (double)RAND_MAX;
            double u2 = (double)rand() / (double)RAND_MAX;

            // compute tau
            double tau = 1/a0 * log(1/u1);

            // find r such that sum(w[0:r]) >= u2 * a0 and sum(w[0:r-1]) < u2 * a0
            double sum = 0;
            int r = 0;
            while (sum < u2 * a0) {
                sum += w[r];
                r++;
            }
            
            // update the state vector x
            for (int j=0; j<7; j++) {
                x[j] += P[(r-1)*7+j];
            }

            // update the current time t
            t += tau;
            if (t >= 25 && t <= 26) {
                one_quarter_time = MPI_Wtime() - start_time;
            }

            if (t >= 50 && t <= 51) {
                half_time = MPI_Wtime() - start_time;
            }

            if (t >= 75 && t <= 76) {
                three_quarter_time = MPI_Wtime() - start_time;
            }
        }

        final_time = MPI_Wtime() - start_time;

        // store the final state vector x
        for (int j=0; j<7; j++) {
            local_final_matrix[j*local_experiments+i] = x[j]; // inefficient way to store the matrix
        }

    }

    // find the local maximum value of the first row
    int local_max = 0;
    for (int i=0; i<local_experiments; i++) {
        if (local_final_matrix[i] > local_max) {
            local_max = local_final_matrix[i];
        }
    }

    // find the local minimum valuse of the first row
    int local_min = local_final_matrix[0];
    for (int i=0; i<local_experiments; i++) {
        if (local_final_matrix[i] < local_min) {
            local_min = local_final_matrix[i];
        }
    }

    // find the global maximum value of the first row
    int global_max = 0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // find the global minimum value of the first row
    int global_min = 0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // find the global bin width
    double bin_width = (global_max - global_min) / bins;

    // for each bin, set the upper bound and lower bound
    int binrange[bins+1];
    for (int i=0; i<bins+1; i++) {
        binrange[i] = global_min + i * (int)bin_width;
    }

    // for each bin, count the number of elements in the first row that fall into the bin
    int *bincount = (int *)calloc(bins, sizeof(int));
    for (int i=0; i<local_experiments; i++) {
        for (int j=0; j<bins; j++) {
            if (local_final_matrix[i] >= binrange[j] && local_final_matrix[i] < binrange[j+1]) {
                bincount[j]++;
            }
        }
    }

    // collect all the bin counts
    int *final_bincount;
    if (rank == 0) {
        final_bincount = (int *)malloc(bins * sizeof(int));
    }
    MPI_Reduce(bincount, final_bincount, bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    // stop timer
    double time_spent = MPI_Wtime() - start_time;
    double max_time_spent = 0;
    MPI_Reduce(&time_spent, &max_time_spent, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // print the quarter time spent
    printf("I am process %d, quarter time spent: %f\n", rank, one_quarter_time);
    printf("I am process %d, half time spent: %f\n", rank, half_time);
    printf("I am process %d, three quarter time spent: %f\n", rank, three_quarter_time);
    printf("I am process %d, final time spent: %f\n", rank, final_time);

    // write the final bin counts to the output file
    if (rank == 0) {
        // print the time spent
        printf("Time spent: %f\n", max_time_spent);

        FILE *fp = fopen(output_file, "w");
        for (int i=0; i<bins; i++) {
            fprintf(fp, "%d - %d: %d\n", binrange[i], binrange[i+1], final_bincount[i]);
        }
        fclose(fp);
        free(final_bincount);
    }

    free(w);
    free(local_final_matrix);
    free(P);
    free(bincount);

    MPI_Finalize();
    return 0;
}

// void 