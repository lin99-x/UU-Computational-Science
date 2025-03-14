/**********************************************************************
 * This program calculates pi using C
 *
 **********************************************************************/
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

const int intervals = 500000000;
double dx  = 1.0/intervals;

struct array {
  int input[2];
};

void* the_pthread_func(void* arg) {
  struct array* area = (struct array*)arg;
  int start = area->input[0];
  int end = area->input[1];
  double x, *sum;
  sum = (double*)malloc(sizeof(double));
  *sum = 0.0;
  for (int i = start; i < end; i++) { 
    x = dx*(i - 0.5);
    *sum += dx*4.0/(1.0 + x*x);
  }
  pthread_exit((void*) sum);
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    printf("Usage: %s N\n", argv[0]);
    return -1;
  }

  int i;
  double sum=0.0, x;
  double* address;
  struct array input;
  double slice = intervals / (atoi(argv[1]) + 1);

  pthread_t threads[atoi(argv[1])];
  for (int t=0; t<atoi(argv[1]); t++) {
    input.input[0] = t * slice;
    input.input[1] = (t + 1) * slice;
    pthread_create(&threads[t], NULL, the_pthread_func, &input);
  }

  for (i = atoi(argv[1]) * slice; i <= intervals; i++) { 
    x = dx*(i - 0.5);
    sum += dx*4.0/(1.0 + x*x);
  }

  // join the threads
  for (int t=0; t<atoi(argv[1]); t++) {
    pthread_join(threads[t], (void*) &address);
    sum = sum + *address;
    free(address);
  }

  printf("PI is approx. %.16f\n",  sum);

  return 0;
}


// serial result: PI is approx. 3.1415926535899894
// parallel result: PI is approx. 2.9551680823720119

// The difference is too big...