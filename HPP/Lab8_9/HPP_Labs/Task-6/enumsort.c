/**********************************************************************
 * Enumeration sort
 *
 **********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define NUM_THREADS	5
#define len 100000
#define Task_size (len / NUM_THREADS)

static double get_wall_seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

double indata[len], outdata[len];

void *findrank(void *arg)
{
	int rank,i,start,end;
	long j=(long)arg;

  // each thread calculate the rank of n numbers, n = len/num_threads
  start = j*Task_size;
  end = (j+1)*Task_size - 1;

  for (i=start; i<=end; i++) {
    rank = 0;
    for (int k=0; k<len; k++) {
      if (indata[k] < indata[i]) {
        rank++;
      }
    }
    outdata[rank] = indata[i];
  }
	
	pthread_exit(NULL);
}


int main(int argc, char *argv[]) {
	
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  int i, j, t;
  long el;
  void *status;
  
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);	

  // Generate random numbers
  for (i=0;i<len;i++){
      indata[i]=drand48();
      outdata[i]=-1.0;
  }

  // Enumeration sort
  double startTime = get_wall_seconds();

  // give the tasks to different threads
  for (j=0; j<NUM_THREADS; j++) {
    el = j;
    pthread_create(&threads[j], &attr, findrank, (void*) el);
  }

  for (t=0; t<NUM_THREADS; t++) {
    pthread_join(threads[t], &status);
  }
  double timeTaken = get_wall_seconds() - startTime;
  printf("Time: %f  NUM_THREADS: %d\n", timeTaken, NUM_THREADS);

  // Check results, -1 implies data same as the previous element
    for (i=0; i<len-1; i++)
      if (outdata[i]>outdata[i+1] && outdata[i+1]>-1)
	printf("ERROR: %f,%f\n", outdata[i],outdata[i+1]);

  return 0;
}
