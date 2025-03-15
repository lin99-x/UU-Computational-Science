#include "stencil.h"


int main(int argc, char **argv) {
	if (4 != argc) {
		printf("Usage: stencil input_file output_file number_of_applications\n");
		return 1;
	}

	
	// Stencil values
	double h = 2.0*PI/num_values;
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

	int num_steps = atoi(argv[3]);
	int rank, size;
	MPI_Init(&argc, &argv); /* Initialize MPI */
  	MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get the number of processors */
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get my number                */

	// 分配内存（局部的input和output）
	
	if (rank==0){
		char *input_name = argv[1];
		char *output_name = argv[2];

		// Read input file
		double *input;
		int num_values;
		if (0 > (num_values = read_input(input_name, &input))) {
			return 2;
		}

		// Allocate data for result
		double *output;
		if (NULL == (output = malloc(num_values * sizeof(double)))) {
			perror("Couldn't allocate memory for output");
			return 2;
		}

		// 发送数据
		
	}

	// 多处变量名需要修改
	// 接收process 0 数据
	
	// Repeatedly apply stencil
	for (int s=0; s<num_steps; s++) {
		// 发送数据给相邻process
		// 获取相邻数据
		
		// Apply stencil
		for (int i=0; i<EXTENT; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j + num_values) % num_values;
				result += STENCIL[j] * input[index];
			}
			output[i] = result;
		}
		for (int i=EXTENT; i<num_values-EXTENT; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = i - EXTENT + j;
				result += STENCIL[j] * input[index];
			}
			output[i] = result;
		}
		for (int i=num_values-EXTENT; i<num_values; i++) {
			double result = 0;
			for (int j=0; j<STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j) % num_values;
				result += STENCIL[j] * input[index];
			}
			output[i] = result;
		}
		// 等待其它process都算到这一步（似乎可以省略？）
		// Swap input and output 
		if (s < num_steps-1) {
			double *tmp = input;
			input = output;
			output = tmp;
		}
		// 等待其它process都算到这一步
	}

	//向process 0 返回数据
	

	if (rank==0){
		// 接收数据
		// Write result
	#ifdef PRODUCE_OUTPUT_FILE
		if (0 != write_output(output_name, output, num_values)) {
			return 2;
		}
	#endif

		// Clean up
		free(input);
		free(output);
	}

	//清除各process内存

	/* 插入合适位置
	// Start timer
	double start = MPI_Wtime();
	// Stop timer
	double my_execution_time = MPI_Wtime() - start;
	// Write result
	printf("%f\n", my_execution_time);
	*/

	MPI_Finalize(); /* Shut down and clean up MPI */
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
