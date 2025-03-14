#include <stdio.h>

int main() {
    double epsilon = 0.001, test;
    for (int i = 0; i < 50; i++) {
        epsilon = epsilon * 0.5;
        test = epsilon + 1;
        if (test == 1) {
            printf("Now epsilon is: %e", epsilon);
        }
    }
    return 0;
}