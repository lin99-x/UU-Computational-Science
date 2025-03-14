#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>

static double get_wall_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
    return seconds;
}

struct data {
    int input[2];
    int result;
};

bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i < num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

void* the_thread_func(void* arg) {
    struct data *thread_data = (struct data *)arg;
    int num_start = thread_data->input[0];
    int num_end = thread_data->input[1];
    int count = 0;
    for (int i = num_start; i < num_end; i++) {
        if (is_prime(i)) count++;
    }
    thread_data->result = count;
    return NULL;
}

int main(int argc, char* argv[]) {
    // check if the given arguments are correct, or it will give segmentation fault
    if (argc != 3) {
        printf("Usage: %s M N\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    long N = atoi(argv[2]);

    // divied by threads
    int ll = M / (N + 1);

    // why am I must allocate memory for thread_data here, in task4 I didn't do it but also works.
    struct data *thread_data = malloc(N * sizeof(struct data));
    if (!thread_data) {
        printf("Error allocating memory for thread data\n");
        return 1;
    }

    double starttime = get_wall_seconds();
    // start the thread
    pthread_t threads[N];
    for (long i = 0; i < N; i++) {
        printf("In main: creating thread %ld\n", i);
        thread_data[i].input[0] = i * ll;  // ll = M / (N + 1)
        thread_data[i].input[1] = (i + 1) * ll;
        int rc = pthread_create(&threads[i], NULL, the_thread_func, &thread_data[i]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            free(thread_data);
            return 1;
        }
    }

    // join the thread
    for (long i = 0; i < N; i++) {
        int rc = pthread_join(threads[i], NULL);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            free(thread_data);
            return 1;
        }
    }

    // sum up the results
    int totalSum = 0;
    for (long i = 0; i < N; i++) {
        totalSum += thread_data[i].result;
    }

    // handle the remainder in the main function
    for (int i = N * ll; i < M; i++) {
        if (is_prime(i)) totalSum++;
    }

    printf("Number of prime numbers: %d\n", totalSum);
    double takentime = get_wall_seconds() - starttime;
    printf("The total time taken is: %lf\n", takentime);
    free(thread_data);
    return 0;
}


// M = 1000000

// Number of threads not include main!

// In main: creating thread 0
// Number of prime numbers: 78498
// The total time taken is: 46.972258


// number of threads:2
// Number of prime numbers: 78498
// The total time taken is: 41.391087


// number of threads: 4
// Number of prime numbers: 78498
// The total time taken is: 29.751735

// number of threads: 6
// Number of prime numbers: 78498
// The total time taken is: 22.874601

// number of threads: 8
// Number of prime numbers: 78498
// The total time taken is: 18.431346

// number of threads: 10
// Number of prime numbers: 78498
// The total time taken is: 15.636894

// number of threads:12
// Number of prime numbers: 78498
// The total time taken is: 13.821524

// number of threads: 14
// Number of prime numbers: 78498
// The total time taken is: 12.573306

// number of threads: 30
// Number of prime numbers: 78498
// The total time taken is: 8.860233

// number of threads: 300
// Number of prime numbers: 78498
// The total time taken is: 6.243270

// running your program with larger numbers of threads, more than the
// number of cores available, no further time improvment.