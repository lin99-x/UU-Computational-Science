#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

long long int find_median (int *data, int len);
// long long int find_mean (int *data, int len);
int compare (const void *a, const void *b);
int* merge(int* a, int* b, int length_a, int length_b);
int pivot_selection(int strategy, int group_rank, int group_size, int *local_array, int local_length, MPI_Comm comm);
bool check_order(int *arr, int size);

int main(int argc, char **argv)
{
    // check if the number of arguments is correct
    if (argc != 4) {
        printf("Usage: %s <input file> <output file> <pivot strategies>\n", argv[0]);
        return -1;
    }

    char *inputfile = argv[1];
    char *outputfile = argv[2];
    int pivot_strategy = atoi(argv[3]);

    FILE *fp;

    // init mpi
    int rank, size, N;
    int *data, *local_array;
    int *final_array;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request request;

    int local_length;
    int rem;
    int sum = 0, sum_final = 0;
    int *count, *displs, *count_final, *displs_final;

    // rank 0 read input file
    if (rank == 0) {
        fp = fopen(inputfile, "r");
        if (fp == NULL) {
            perror("Error opening input file");
            return -1;
        }

        // read the number of elements to be sorted
        if (EOF == fscanf(fp, "%d", &N)) {
            perror("Error when reading number of elements from input file");
            return -1;
        }

        data = (int *)malloc(N * sizeof(int));
    
        // read the data from input file
        for (int i = 0; i < N; i++) {
            if (EOF == fscanf(fp, "%d", &data[i])) {
                perror("Error when reading data from input file");
                return -1;
            }
        }

        fclose(fp);
        // printf("data: ");
        // for (int i = 0; i < N; i++) {
        //     printf("%d  ", data[i]);
        // }
        // printf("\n");
    }

    double start_time = MPI_Wtime(); // start timing
    
    // Algorithm step 1
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast N to all processes

    rem = N % size; // remainder
    count = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    // calculate the number of elements to be sent to each process
    for (int i=0; i<size; i++) {
        count[i] = N / size;
        if (rem != 0) {
            count[i] += 1;
            rem--;
        }

        displs[i] = sum;
        sum += count[i];
    }

    local_length = count[rank];
    local_array = (int *)malloc(local_length * sizeof(int));

    // scatter the array to all processes
    MPI_Scatterv(data, count, displs, MPI_INT, local_array, local_length, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(data);
    }

    // Algorithm step 2
    // sort the local array
    qsort(local_array, local_length, sizeof(int), compare); 

    // Algorithm step 3 and 4
    int group_rank = rank;
    int group_size = size;
    int pivot;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm new_comm;

    // printf("before while----------------------------------------------\n");
    // printf("%d: ", rank);
    // for (int i = 0; i < count[rank]; i++) {
    //     printf("%d\t", local_array[i]);
    // }
    // printf("\n");


    if (size > 1) {
        while (group_size > 1) {

            pivot = pivot_selection(pivot_strategy, group_rank, group_size, local_array, local_length, comm);
            MPI_Bcast(&pivot, 1, MPI_INT, 0, comm);

            // printf("rank: %d, pivot: %d\n", group_rank, pivot);


            int division_index = local_length / 2;
            for (; (division_index > 0) && local_array[division_index] >= pivot; division_index--){}
            for (; (division_index < local_length) && local_array[division_index] < pivot; division_index++){}

            // printf("rank: %d, division_index: %d\n", group_rank, division_index);

            int small = division_index;
            int equal_large = local_length - small;

            // printf("rank: %d, small: %d, equal_large: %d\n", group_rank, small, equal_large);

            int pair_rank;
            if (group_rank < group_size / 2) {
                pair_rank = group_rank + group_size / 2;
            }
            else {
                pair_rank = group_rank - group_size / 2;
            }

            int recv_small = 0, recv_equal_large = 0;
            int *recv_array, *kept_array;

            // send the number of elements to be sent to the pair process
            if (group_rank < group_size / 2) {
                //MPI_Isend(&equal_large, 1, MPI_INT, pair_rank, 0, comm, &request);
                //MPI_Irecv(&recv_small, 1, MPI_INT, pair_rank, 0, comm, &request);
		        MPI_Sendrecv(&equal_large, 1, MPI_INT, pair_rank, 0, &recv_small, 1, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            }
            else {
                //MPI_Isend(&small, 1, MPI_INT, pair_rank, 0, comm, &request);
                //MPI_Irecv(&recv_equal_large, 1, MPI_INT, pair_rank, 0, comm, &request);
		        MPI_Sendrecv(&small, 1, MPI_INT, pair_rank, 0, &recv_equal_large, 1, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            }
	        //MPI_Wait(&request, MPI_STATUS_IGNORE);
            //MPI_Barrier(comm);
            // can try to use unblocking send and receive
            // if (group_rank < group_size / 2) {
            //     recv_array = (int *)malloc(recv_small * sizeof(int));
            //     MPI_Send(local_array + small, equal_large, MPI_INT, pair_rank, 0, comm);
            //     MPI_Recv(recv_array, recv_small, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            // }
            // else {
            //     recv_array = (int *)malloc(recv_equal_large * sizeof(int));
            //     MPI_Send(local_array, small, MPI_INT, pair_rank, 0, comm);
            //     MPI_Recv(recv_array, recv_equal_large, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            // }

            // printf("before send and receive----------------------------------------------\n");
            // printf("%d: ", group_rank);
            // printf("recv_small: %d, recv_equal_large: %d\n", recv_small, recv_equal_large);

            // use unblocking send and receive (if not it will be super slow)
            if (group_rank < group_size / 2) {
                recv_array = (int *)malloc(recv_small * sizeof(int));
                //MPI_Isend(local_array + small, equal_large, MPI_INT, pair_rank, 0, comm, &request);
                //MPI_Irecv(recv_array, recv_small, MPI_INT, pair_rank, 0, comm, &request);
		        MPI_Sendrecv(local_array + small, equal_large, MPI_INT, pair_rank, 0, recv_array, recv_small, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            }
            else {
                recv_array = (int *)malloc(recv_equal_large * sizeof(int));
                //MPI_Isend(local_array, small, MPI_INT, pair_rank, 0, comm, &request);
                //MPI_Irecv(recv_array, recv_equal_large, MPI_INT, pair_rank, 0, comm, &request);
		        MPI_Sendrecv(local_array, small, MPI_INT, pair_rank, 0, recv_array, recv_equal_large, MPI_INT, pair_rank, 0, comm, MPI_STATUS_IGNORE);
            }
	        //MPI_Wait(&request, MPI_STATUS_IGNORE);
            //MPI_Barrier(comm); // wait for the send and receive to finish

            // merge two sets of numbers
            if (group_rank < group_size / 2) {
		        kept_array = (int*) malloc(small*sizeof(int));
		        memcpy(kept_array, local_array, small*sizeof(int));
		        free(local_array);
                local_length = small + recv_small;
                local_array = merge(kept_array, recv_array, small, recv_small);
            }
            else {
		        kept_array = (int*) malloc(equal_large*sizeof(int));
		        memcpy(kept_array, local_array + small, equal_large*sizeof(int));
		        free(local_array);
                local_length = equal_large + recv_equal_large;
                local_array = merge(kept_array, recv_array, equal_large, recv_equal_large);
            }

            //MPI_Barrier(comm);
            free(recv_array);
	        free(kept_array);

            MPI_Comm_split(comm, group_rank < group_size / 2, group_rank, &new_comm);
            MPI_Comm_rank(new_comm, &group_rank);
            MPI_Comm_size(new_comm, &group_size);
            comm = new_comm;
        }

        count_final = (int *)malloc(size * sizeof(int));
        displs_final = (int *)malloc(size * sizeof(int));
        // count_final[rank] = local_length;

        MPI_Allgather(&local_length, 1, MPI_INT, count_final, 1, MPI_INT, MPI_COMM_WORLD);
        for (int i=0; i<size; i++) {
            displs_final[i] = sum_final;
            sum_final += count_final[i];
        }

        if (rank == 0) {
            final_array = (int *)malloc(N * sizeof(int));
        }


        MPI_Gatherv(local_array, local_length, MPI_INT, final_array, count_final, displs_final, MPI_INT, 0, MPI_COMM_WORLD);
        
    }

    else {
        final_array = local_array;
    }

    double end_time = MPI_Wtime();
    double run_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&run_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("%f\n", max_time);
        int check = 1;
        for (int i=0; i<N-1; i++) {
            if (final_array[i] > final_array[i+1]) {
                printf("Final_array[%d] is %d, final_array[%d] is %d\n", i, final_array[i], i+1, final_array[i+1]);
                check = 0;
            }
        }
        if (check == 1) {
            printf("The result is correct!\n");
        }

        bool sorted;
        sorted = check_order(final_array, N);
        printf("result %d\n", sorted);
        // for (int i=0; i<N; i++){
        //     printf("%d  ", final_array[i]);
        // }


        // // write out the result
        // FILE *output = NULL;
        // if (NULL == (output = fopen(outputfile, "w"))) {
        //     printf("Error opening output file!\n");
        //     return -1;
        // }

        // for (int i=0; i<N; i++) {
        //     fprintf(output, "%d ", final_array[i]);
        // }
        
        // fclose(output);

        free(final_array);
    }

    free(count);
    free(displs);
    free(count_final);
    free(displs_final);
    free(local_array);

    MPI_Finalize();
    
    return 0;
}

int pivot_selection(int strategy, int group_rank, int group_size, int *local_array, int local_length, MPI_Comm comm) {
    int pivot;
    switch(strategy){
        // Select the median in one processor in each group of processors.
        case 1:
            if (group_rank == 0){
                // printf("group rank 0 has %d elements.", local_length);
                pivot = find_median(local_array, local_length);
                // printf("in pivot_selection function, pivot is: %d", pivot);
            }
        break;
        // Select the median of all medians in each processor group.
        case 2:
        {
            int local_pivot = find_median(local_array, local_length);
            int all_pivots[group_size];
            MPI_Gather(&local_pivot, 1, MPI_INT, all_pivots, 1, MPI_INT, 0, comm);
            if (group_rank == 0){
                qsort(all_pivots, group_size, sizeof(int), compare);
                pivot = find_median(all_pivots, group_size);
            }
        }
        break;
        // Select the mean value of all medians in each processor group.
        case 3: 
        {
            int local_pivot = find_median(local_array, local_length);
            // int all_pivots[group_size];
            long long int pivot;
            // there might be a overflow
            MPI_Reduce(&local_pivot, &pivot, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);
            // MPI_Gather(&local_pivot, 1, MPI_INT, all_pivots, 1, MPI_INT, 0, comm);
            // if (group_rank == 0) pivot = find_mean(all_pivots, group_size);
            // reduce one function call
            pivot = pivot / group_size;
            // printf("pivot is %d\n", pivot);
        }
        break;
    }
    return pivot;
}

int* merge(int* a, int* b, int length_a, int length_b){
    int len = length_a + length_b;
    int* array = (int *)malloc(len * sizeof(int));
    for (int i = 0, j = 0, k = 0; k < len;) {
        if (i < length_a && (j >= length_b || a[i] < b[j])) {
            array[k++] = a[i++];
        } else {
            array[k++] = b[j++];
        }
    }
    return array;
}


/* define the compare function */
int compare (const void *a, const void *b){
    // the sort is in ascending order
    return ( *(int*)a - *(int*)b );
}

/* define pivot strategies */ 
long long int find_median (int *data, int len)
{
    if (len == 0) {
        return 0;
    }
    if (len == 1) {
        return data[0];
    }
    // if length is odd then return the middle element
    if (len % 2 == 1) {
        return data[len / 2];
    }
    long long int sum = (long int)data[len / 2 - 1] + (long int)data[len / 2];
    return sum / 2;
}

// long long int find_mean (int *data, int len)
// {
//     if (len == 0) {
//         return 0;
//     }
//     long long int sum = 0, mean = 0;
//     for (int i = 0; i < len; i++) {
//         sum += data[i];
//     }
//     mean = sum / (int)len;
//     return mean;
// }

bool check_order(int *arr, int size) 
{
    for (int i=1; i<size; i++) {
        if (arr[i] < arr[i-1]) {
            return false;
        }
    }
    return true;
}