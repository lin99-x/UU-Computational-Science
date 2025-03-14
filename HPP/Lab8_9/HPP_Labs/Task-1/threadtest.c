#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

void* the_thread_func(void* arg) {
  int* p = (int*)malloc(2*sizeof(int));
  p[0] = 4;
  p[1] = 9;
  // return p;
  pthread_exit((void*)p);
}

int main() {
  printf("This is the main() function starting.\n");

  /* Start thread. */
  pthread_t thread;
  printf("the main() function now calling pthread_create().\n");
  if(pthread_create(&thread, NULL, the_thread_func, NULL) != 0) {
    printf("ERROR: pthread_create failed.\n");
    return -1;
  }

  printf("This is the main() function after pthread_create()\n");

  int* address;

  /* Wait for thread to finish. */
  printf("the main() function now calling pthread_join().\n");
  if(pthread_join(thread, (void*) &address) != 0) {
    printf("ERROR: pthread_join failed.\n");
    return -1;
  }

  printf("The result from the thread is: %d %d", address[0], address[1]);
  free(address);
  return 0;
}
