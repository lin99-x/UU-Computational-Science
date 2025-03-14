#include <stdio.h>
#include <pthread.h>

void* the_thread_func(void* arg) {
  printf("The thread starts working now! ");
  long int i;
  double sum1;
  for (i=0; i<1000000000; i++)
    sum1 += 1e-9;
  printf("Result of work in thread(): sum = %f\n", sum1);
  return NULL;
}

void* the_thread_func_B(void* arg) {
  printf("The second thread starts working now! ");
  long int i;
  double sum2;
  for (i=0; i<100000000; i++)
    sum2 += 0.1;
  printf("Result of work in thread2(): sum = %f\n", sum2);
  return NULL;
}

int main() {
  printf("This is the main() function starting.\n");

  /* Start thread. */
  pthread_t thread;
  pthread_t threadB;
  printf("the main() function now calling pthread_create().\n");
  pthread_create(&thread, NULL, the_thread_func, NULL);
  pthread_create(&thread, NULL, the_thread_func_B, NULL);

  printf("This is the main() function after pthread_create()\n");

  printf("main() starting doing some work.\n");
  long int i;
  double sum;
  for (i=0; i<1000000000; i++)
    sum += 1e-7;
  printf("Result of work in main(): sum = %f\n", sum);

  /* Wait for thread to finish. */
  printf("the main() function now calling pthread_join().\n");
  pthread_join(thread, NULL);
  pthread_join(threadB, NULL);

  printf("Now the work is done. ");
  return 0;
}


// When I use two threads, the %CPU is 200, but when I use three threads, the %CPU is also 200, is this because my third thread has too 
// less things to do or I just have two cores.