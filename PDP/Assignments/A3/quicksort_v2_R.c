#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int cmpfunc (const void * a, const void * b);
void recursive_global_sort(int** local_array, int* local_length, int pivot_strategy, MPI_Comm comm);
void pivot_selection(int* pivot, int pivot_strategy, int *local_array, int local_length, MPI_Comm comm);
int find_median(int *data, int len);
int* merge(int* a, int* b, int length_a, int length_b);

int main(int argc, char **argv){
    // check if the number of arguments is correct
    if (argc != 4) {
        printf("Usage: %s <input file> <output file> <pivot strategies>\n", argv[0]);
        return -1;
    }

    char *inputfile = argv[1];
    char *outputfile = argv[2];
    int pivot_strategy = atoi(argv[3]);
    FILE *fp;
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int* data; 
    int N;
    int rem, sum;
    int sum_final;
    int count[size], displs[size];
    int count_final[size], displs_final[size];
    int local_length;
    int* local_array;
    int* final_array;
    double max_time = 1; //for self test
    
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
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rem = N % size; // remainder
    sum = 0;
    for (int i=0; i<size; i++){
        count[i] = N / size;
        if (i<rem) count[i]++;
        displs[i] = sum;
        sum += count[i];
    }

    local_length = count[rank];
    local_array = (int *)malloc(local_length * sizeof(int));
    MPI_Scatterv(data, count, displs, MPI_INT, local_array, local_length, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) free(data);

    qsort(local_array, local_length, sizeof(int), cmpfunc);
    recursive_global_sort(&local_array, &local_length, pivot_strategy, MPI_COMM_WORLD);
    MPI_Allgather(&local_length, 1, MPI_INT, count_final, 1, MPI_INT, MPI_COMM_WORLD);

    sum_final = 0;
    for (int i=0; i<size; i++) {
        displs_final[i] = sum_final;
        sum_final += count_final[i];
    }

    if (rank == 0) final_array = (int *)malloc(N * sizeof(int));
    MPI_Gatherv(local_array, local_length, MPI_INT, final_array, count_final, displs_final, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f\n", max_time);
        //write out the result
        FILE *output = NULL;
        if (NULL == (output = fopen(outputfile, "w"))) {
            printf("Error opening output file!\n");
            return -1;
        }
        for (int i=0; i<N; i++) {
            fprintf(output, "%d ", final_array[i]);
        }
        fclose(output);
        free(final_array);
    }
    
    free(local_array);
    MPI_Finalize();
    return 0;
}

void recursive_global_sort(int** local_array, int* local_length, int pivot_strategy, MPI_Comm comm){
    int rank, size;
    int pivot;
    int length;
    int large, small;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Request request;
    MPI_Comm halved_communicator;
    MPI_Allreduce(local_length, &length, 1, MPI_INT, MPI_SUM, comm);
    
    if (size==1 || length==0) return;
    else{
        pivot_selection(&pivot, pivot_strategy, *local_array, *local_length, comm);
        if (*local_length != 0){
            int division_index = *local_length / 2;
            // to make workload balanced, we do not need to find the smallest element that is equal to or larger than pivot
            // when we have find a division_index such that local_array[division_index] == pivot
            if ((*local_array)[division_index] != pivot){
                for (; (division_index > 0) && (*local_array)[division_index] >= pivot; division_index--){}
                for (; (division_index < *local_length) && (*local_array)[division_index] < pivot; division_index++){}
            }
            small = division_index;
            large = *local_length - small;
        }
        else {
            small = 0;
            large = 0;
        }

        int pair_rank;
        if (rank < size / 2) pair_rank = rank + size / 2;
        else pair_rank = rank - size / 2;
        int recv_small = 0, recv_large = 0;
        int *recv_array;
        int *kept_array;

        // send the number of elements to be sent to the pair process
        if (rank < size / 2) {
            MPI_Isend(&large, 1, MPI_INT, pair_rank, 0, comm, &request);
            MPI_Irecv(&recv_small, 1, MPI_INT, pair_rank, 0, comm, &request);
        }
        else {
            MPI_Isend(&small, 1, MPI_INT, pair_rank, 0, comm, &request);
            MPI_Irecv(&recv_large, 1, MPI_INT, pair_rank, 0, comm, &request);
        }
        MPI_Barrier(comm); //MPI_Wait?
        
        if (rank < size / 2) {
            recv_array = (int *)malloc(recv_small * sizeof(int));
            MPI_Isend(*local_array + small, large, MPI_INT, pair_rank, 0, comm, &request);
            MPI_Irecv(recv_array, recv_small, MPI_INT, pair_rank, 0, comm, &request);
            kept_array = (int* )malloc(small*sizeof(int));
            memcpy(kept_array, *local_array, small*sizeof(int));
        
        }
        else {
            recv_array = (int *)malloc(recv_large * sizeof(int));
            MPI_Isend(*local_array, small, MPI_INT, pair_rank, 0, comm, &request);
            MPI_Irecv(recv_array, recv_large, MPI_INT, pair_rank, 0, comm, &request);
            kept_array = (int* )malloc(large*sizeof(int));
            memcpy(kept_array, *local_array + small, large*sizeof(int));
        }
        free(*local_array);
        MPI_Barrier(comm); // wait for the send and receive to finish
        
        // merge two sets of numbers
        if (rank < size / 2) {
            *local_length = small + recv_small;
            *local_array = merge(kept_array, recv_array, small, recv_small);
        }
        else {
            *local_length = large + recv_large;
            *local_array = merge(kept_array, recv_array, large, recv_large);
        }
        free(recv_array);
        free(kept_array);
        MPI_Barrier(comm);
        

        int color = (rank >= size/2);
        MPI_Comm_split(comm, color, rank, &halved_communicator);
        recursive_global_sort(local_array, local_length, pivot_strategy, halved_communicator);
        MPI_Comm_free(&halved_communicator);
        return;
    }
    
}

int cmpfunc(const void * a, const void * b){
    return ( *(int*)a - *(int*)b );
}

void pivot_selection(int* pivot, int pivot_strategy, int *local_array, int local_length, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    switch(pivot_strategy){
        // Select the median in one processor in each group of processors.
        case 1:
        {
            if (rank == 0){
                // printf("group rank 0 has %d elements.", local_length);
                *pivot = find_median(local_array, local_length);
                // printf("in pivot_selection function, pivot is: %d", pivot);
            }            
            MPI_Bcast(pivot, 1, MPI_INT, 0, comm);
        }
        break;
        // Select the median of all medians in each processor group.
        case 2:
        {   
            *pivot = find_median(local_array, local_length);
            int all_pivots[size];
            MPI_Gather(pivot, 1, MPI_INT, all_pivots, 1, MPI_INT, 0, comm);
            if (rank == 0){
                qsort(all_pivots, size, sizeof(int), cmpfunc);
                *pivot = find_median(all_pivots, size);
            }
            MPI_Bcast(pivot, 1, MPI_INT, 0, comm);
        }
        break;
        // Select the mean value of all medians in each processor group.
        case 3: 
        {
            long long int pivot_sum;
            *pivot = find_median(local_array, local_length);
            MPI_Reduce(pivot, &pivot_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);
            if (rank==0) *pivot = pivot_sum / size;
            MPI_Bcast(pivot, 1, MPI_INT, 0, comm);
        }
        break;
    }
    return;
}

int find_median(int *data, int len){
    // data must be sorted in advance
    if (len==0) return 0;
    else if (len % 2 == 1) return data[len / 2]; // if length is odd then return the middle element
    return (data[len / 2 - 1] + data[len / 2]) / 2;
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
