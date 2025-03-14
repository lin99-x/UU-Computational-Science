#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>

struct data {
    int input;
    int result;
};

bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i=2; i<num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

void* the_thread_func(void* arg) {
    struct data *thread_data = (struct data *)arg;
    int num = thread_data->input;
    int count = 0;
    for (int i=1; i<num; i++) {
        if (is_prime(i)) count++;
    }
    thread_data->result = count;
    return NULL;
}

int main() {
    int l, ll;
    int count = 0;

    printf("Give a number to create a test list: ");
    scanf("%d", &l);

    // even number
    if (l % 2 == 0) {
        ll = l/2;
    }
    else {
        ll = (l+1)/2;
    }

    struct data thread_data;
    thread_data.input = l - ll;

    // start the thread
    pthread_t thread;
    pthread_create(&thread, NULL, the_thread_func, &thread_data);
    for (int i=ll; i<l; i++) {
        if (is_prime(i)) count++;
    }

    // join the thread
    pthread_join(thread, NULL);
    printf("sum computed by main() : %d\n", count);
    printf("sum computed by thread : %d\n", thread_data.result);
    int totalSum = count + thread_data.result;
    printf("Number of prime numbers: %d\n", totalSum);
    return 0;
}