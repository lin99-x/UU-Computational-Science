/*****************************************************************************
 * FILE: join.c
 *
 ******************************************************************************/
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define NUM_THREADS	4

void *BusyWork(void *t)
{
  int i;
  long tid;
  double result=0.0;
  tid = (long)t;
  printf("Thread %ld starting...\n",tid);
  for (i=0; i<20000000; i++)
    {
      result = result + sin(i) * tan(i);
    }
  printf("Thread %ld done. Result = %e\n",tid, result);
  pthread_exit((void*) t);
}

int main (int argc, char *argv[])
{
  pthread_t thread[NUM_THREADS];
  pthread_attr_t attr;
  int rc;
  long t;
  // void *status;

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  for(t=0; t<NUM_THREADS; t++) {
    printf("Main: creating thread %ld\n", t);
    rc = pthread_create(&thread[t], &attr, BusyWork, (void *)t); 
    if (rc) {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  /* Free attribute and wait for the other threads */
  pthread_attr_destroy(&attr);
  // for(t=0; t<NUM_THREADS; t++) {
  //   rc = pthread_join(thread[t], &status);
  //   if (rc) {
  //     printf("ERROR; return code from pthread_join() is %d\n", rc);
  //     exit(-1);
  //   }
  //   printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
  // }
 
  printf("Main: program completed. Exiting.\n");
  // pthread_exit(NULL);
}

// if remove the pthread_exit in the end of main(), there is no result 
// displayed, cause the the results cannot be passed.

// detached
// Main: creating thread 0
// Main: creating thread 1
// Main: creating thread 2
// Main: creating thread 3
// Main: program completed. Exiting.
// Thread 0 starting...
// Thread 1 starting...
// Thread 2 starting...
// Thread 3 starting...
// Thread 1 done. Result = 4.931540e+06
// Thread 0 done. Result = 4.931540e+06
// Thread 3 done. Result = 4.931540e+06
// Thread 2 done. Result = 4.931540e+06

// joined
// Main: creating thread 0
// Main: creating thread 1
// Main: creating thread 2
// Main: creating thread 3
// Thread 0 starting...
// Thread 1 starting...
// Thread 3 starting...
// Thread 2 starting...
// Thread 0 done. Result = 4.931540e+06
// Main: completed join with thread 0 having a status of 0
// Thread 1 done. Result = 4.931540e+06
// Main: completed join with thread 1 having a status of 1
// Thread 3 done. Result = 4.931540e+06
// Thread 2 done. Result = 4.931540e+06
// Main: completed join with thread 2 having a status of 2
// Main: completed join with thread 3 having a status of 3
// Main: program completed. Exiting.