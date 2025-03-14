#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>

const long int N1 = 300000000;
const long int N2 = 500000000;

static double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
    return seconds;
}

void* the_thread_func(void* arg) {
  long int i;
  long int sum = 0;
  for(i = 0; i < N2; i++)
    sum += 7;
  /* OK, now we have computed sum. Now copy the result to the location given by arg. */
  long int * resultPtr;
  resultPtr = (long int *)arg;
  *resultPtr = sum;
  return NULL;
}

int main() {
  printf("This is the main() function starting.\n");
  double starttime = get_wall_seconds();
  long int thread_result_value = 0;

  /* Start thread. */
  pthread_t thread;
  printf("the main() function now calling pthread_create().\n");
  pthread_create(&thread, NULL, the_thread_func, &thread_result_value);

  printf("This is the main() function after pthread_create()\n");

  long int i;
  long int sum = 0;
  for(i = 0; i < N1; i++)
    sum += 7;

  /* Wait for thread to finish. */
  printf("the main() function now calling pthread_join().\n");
  pthread_join(thread, NULL);

  printf("sum computed by main() : %ld\n", sum);
  printf("sum computed by thread : %ld\n", thread_result_value);
  long int totalSum = sum + thread_result_value;
  printf("totalSum : %ld\n", totalSum);
  double timetaken = get_wall_seconds() - starttime;
  printf("The total time taken is: %lf\n", timetaken);
  return 0;
}


// N1 = 700000000;
// N2 = 100000000;
// sum computed by main() : 4900000000
// sum computed by thread : 700000000
// totalSum : 5600000000
// The total time taken is: 0.670649


// N1 = 600000000;
// N2 = 200000000;
// sum computed by main() : 4200000000
// sum computed by thread : 1400000000
// totalSum : 5600000000
// The total time taken is: 0.576155


// N1 = 500000000;
// N2 = 300000000;
// sum computed by main() : 3500000000
// sum computed by thread : 2100000000
// totalSum : 5600000000
// The total time taken is: 0.483797


// N1 = 400000000;
// N2 = 400000000;
// sum computed by main() : 2800000000
// sum computed by thread : 2800000000
// totalSum : 5600000000
// The total time taken is: 0.388903

// N1 = 500000000;
// N2 = 300000000;
// sum computed by main() : 2100000000
// sum computed by thread : 3500000000
// totalSum : 5600000000
// The total time taken is: 0.481441