#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_data {
   int  thread_id;
};

void *print_thread_index(void *thread_arg) {
   struct thread_data *my_data;
   my_data = (struct thread_data *) thread_arg;
   printf("Thread %d\n", my_data->thread_id);
   return NULL;
}

int main(int argc, char *argv[]) {
   pthread_t threads[atoi(argv[1])];
   struct thread_data thread_data_array[atoi(argv[1])];
   int rc;
   long t;
   for(t=0; t<atoi(argv[1]); t++){
      printf("In main: creating thread %ld\n", t);
      thread_data_array[t].thread_id = t;
      rc = pthread_create(&threads[t], NULL, print_thread_index, (void *) &thread_data_array[t]);
      if (rc){
         printf("ERROR; return code from pthread_create() is %d\n", rc);
         exit(-1);
      }
   }
   pthread_exit(NULL);
}
