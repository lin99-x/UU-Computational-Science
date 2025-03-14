#include <stdio.h>
#include <pthread.h>

struct data {
    int caller;
    int callee;
};

void* thread_func1(void* arg) {
    struct data* i = (struct data*)arg;
    int which = i->callee;
    int index = i->caller;

    printf("This is the %d subthread called by thread %d\n ", which, index);


    return NULL;
}


void* the_thread_func(void* arg) {
    int* i = (int*)arg;
    struct data index;
    index.caller = *i;
    pthread_t thread;
    pthread_t threadB;
    printf("The thread %d is calling first subthread now.\n ", index.caller);
    index.callee = 1;
    pthread_create(&thread, NULL, thread_func1, &index);

    printf("The thread %d is calling second subthread now.\n ", index.caller);
    index.callee = 2;
    pthread_create(&threadB, NULL, thread_func1, &index);

    printf("The thread %d is calling pthread_join function now.\n ", index.caller);
    pthread_join(thread, NULL);
    pthread_join(threadB, NULL);

    return NULL;
}



int main() {
    int t1 = 1;
    int t2 = 2;
    printf("The main() call pthread1 now.\n ");
    pthread_t thread1;
    pthread_create(&thread1, NULL, the_thread_func, &t1);
    printf("The main() call pthread2 now.\n ");
    pthread_t thread2;
    pthread_create(&thread2, NULL, the_thread_func, &t2);
    // join function
    printf("The main function now calling pthread_join().\n ");
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    return 0;
}


// result
//  The main() call pthread1 now.
//  The main() call pthread2 now.
//  The thread 1 is calling first subthread now.
//  The thread 2 is calling first subthread now.
//  The main function now calling pthread_join().
//  This is the 1 subthread called by thread 1
//  The thread 1 is calling second subthread now.
//  This is the 1 subthread called by thread 2
//  The thread 1 is calling pthread_join function now.
//  The thread 2 is calling second subthread now.
//  This is the 2 subthread called by thread 1
//  The thread 2 is calling pthread_join function now.
//  This is the 2 subthread called by thread 2