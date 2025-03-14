#include "stencil.h"

int main(int argc, char **argv) {
	if (4 != argc) {
		printf("Usage: stencil input_file output_file number_of_applications\n");
		return 1;
	}
	char *input_name = argv[1];
	char *output_name = argv[2];
	int num_steps = atoi(argv[3]); // how many times use stencil

	// Read input file
	double *input;
	int num_values;  // N
	if (0 > (num_values = read_input(input_name, &input))) {
		return 2;
	}

	// Stencil values
	double h = 2.0*PI/num_values;
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

	// Start timer
	double start = MPI_Wtime();

	// Allocate data for final result
	double *output_final;
	if (NULL == (output_final = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for output");
		return 2;
	}

	// MPI values
	int numprocs, myid, chunk;
	MPI_Status status;

	// Assume the total size of array is divisible by number of processors
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	chunk = num_values / numprocs; // assume this is integer, how many elements we send to each processor
	printf("chunk = %d\n", chunk);
	double *localarray = (double *)malloc(chunk * sizeof(double)); // buffer to save array in each processor

	// send a chunk of array to each processor
	MPI_Scatter(input, chunk, MPI_DOUBLE, localarray, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	// Allocate data for result
	double *output;
	if (NULL == (output = malloc(chunk * sizeof(double)))) {
		perror("Couldn't allocate memory for output");
		return 2;
	}

	// Repeatedly apply stencil
	for (int s=0; s<num_steps; s++) {
		double *send_array = (double *)malloc(4 * sizeof(double));
		double *receive_array_left = (double *)malloc(2 * sizeof(double));
		double *receive_array_right = (double *)malloc(2 * sizeof(double));
		send_array[0] = localarray[0];
		send_array[1] = localarray[1];
		send_array[2] = localarray[chunk-2];
		send_array[3] = localarray[chunk-1];
		// printf("%lf %lf %lf %lf\n", send_array[0], send_array[1], send_array[2], send_array[3]);

		// printf("%d\n", myid);
		// communicate with other processors
		if (myid == 0) {
			// printf("myid is %d, %lf %lf %lf %lf\n", myid, send_array[0], send_array[1], send_array[2], send_array[3]);	
			MPI_Send(send_array, 2, MPI_DOUBLE, numprocs-1, 000, MPI_COMM_WORLD);   // send data should on the left side to the last processor
			MPI_Send(&send_array[2], 2, MPI_DOUBLE, myid+1, 000, MPI_COMM_WORLD);   // send data should on the left side
			MPI_Recv(receive_array_left, 2, MPI_DOUBLE, numprocs-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(receive_array_right, 2, MPI_DOUBLE, myid+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			// printf("P0 receive 2 elements on the left side from P%d and the value is %lf %lf.\n", numprocs-1, receive_array_left[0], receive_array_left[1]);
			// printf("P0 receive 2 elements on the right side from P%d and the value is %lf %lf.\n", myid+1, receive_array_right[0], receive_array_right[1]);
		}
		else if (myid == numprocs-1) {
			// printf("myid is %d, %lf %lf %lf %lf\n", myid, send_array[0], send_array[1], send_array[2], send_array[3]);	
			MPI_Send(send_array, 2, MPI_DOUBLE, myid-1, numprocs-1, MPI_COMM_WORLD);
			MPI_Send(&send_array[2], 2, MPI_DOUBLE, 0, numprocs-1, MPI_COMM_WORLD);
			MPI_Recv(receive_array_right, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(receive_array_left, 2, MPI_DOUBLE, myid-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			// printf("P%d receive 2 elements on the left side from p%d and the value is %lf %lf.\n", myid, myid-1, receive_array_left[0], receive_array_left[1]);
			// printf("P%d receive 2 elements on the right side from p%d and the value is %lf %lf.\n", myid, 0, receive_array_right[0], receive_array_right[1]);
		}
		else {
			// printf("myid is %d, %lf %lf %lf %lf\n", myid, send_array[0], send_array[1], send_array[2], send_array[3]);	
			MPI_Send(send_array, 2, MPI_DOUBLE, myid-1, myid, MPI_COMM_WORLD);
			MPI_Send(&send_array[2], 2, MPI_DOUBLE, myid+1, myid, MPI_COMM_WORLD);
			MPI_Recv(receive_array_left, 2, MPI_DOUBLE, myid-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(receive_array_right, 2, MPI_DOUBLE, myid+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			// printf("P%d receive 2 elements on the left side from p%d and the value is %lf %lf.\n", myid, myid-1, receive_array_left[0], receive_array_left[1]);
			// printf("P%d receive 2 elements on the right side from p%d and the value is %lf %lf.\n", myid, myid+1, receive_array_right[0], receive_array_right[1]);
		}

		// Apply stencil
		for (int i=0; i<EXTENT; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j + chunk) % chunk;
				// how to create this?
				result += STENCIL[j] * localarray[index];
			}
			output[i] = result;
			printf("myid is %d, output %d is %lf",myid, i, output[i]);
		}
		for (int i=EXTENT; i<chunk-EXTENT; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = i - EXTENT + j;
				result += STENCIL[j] * localarray[index];
			}
			output[i] = result;
		}
		for (int i=chunk-EXTENT; i<chunk; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j) % chunk;
				result += STENCIL[j] * localarray[index];
			}
			output[i] = result;
		}
		// Swap input and output
		if (s < num_steps-1) {
			double *tmp = input;
			input = output;
			output = tmp;
		}
		free(send_array);
		free(receive_array_left);
		free(receive_array_right);
	}
	free(input);

	// Gather from all processors
	MPI_Gather(&localarray, chunk, MPI_DOUBLE, &output_final, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	free(localarray);
	free(output);

	// Stop timer
	double my_execution_time = MPI_Wtime() - start;

	// Write result
	printf("%f\n", my_execution_time);
#ifdef PRODUCE_OUTPUT_FILE
	if (0 != write_output(output_name, output_final, num_values)) {
		return 2;
	}
#endif

	// Clean up
	MPI_Finalize();
	free(output_final);
	return 0;
}


int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i=0; i<num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)){
		perror("Warning: couldn't close input file");
	}
	return num_values;
}


int write_output(char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}




	// Repeatedly apply stencil
	// for (int s=0; s<num_steps; s++) {
	// 	// Apply stencil
	// 	for (int i=0; i<EXTENT; i++) {
	// 		double result = 0;
	// 		for (int j=0; j<STENCIL_WIDTH; j++) {
	// 			int index = (i - EXTENT + j + num_values) % num_values;
	// 			result += STENCIL[j] * input[index];
	// 		}
	// 		output[i] = result;
	// 	}
	// 	for (int i=EXTENT; i<num_values-EXTENT; i++) {
	// 		double result = 0;
	// 		for (int j=0; j<STENCIL_WIDTH; j++) {
	// 			int index = i - EXTENT + j;
	// 			result += STENCIL[j] * input[index];
	// 		}
	// 		output[i] = result;
	// 	}
	// 	for (int i=num_values-EXTENT; i<num_values; i++) {
	// 		double result = 0;
	// 		for (int j=0; j<STENCIL_WIDTH; j++) {
	// 			int index = (i - EXTENT + j) % num_values;
	// 			result += STENCIL[j] * input[index];
	// 		}
	// 		output[i] = result;
	// 	}
	// 	// Swap input and output
	// 	if (s < num_steps-1) {
	// 		double *tmp = input;
	// 		input = output;
	// 		output = tmp;
	// 	}
	// }