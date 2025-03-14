#include <stdio.h>

int main() {
    // variables
    double i = 5.6;
    char c = 'a';
    int j = 4;
    // pointers
    double *dPtr = &i;
    char *chPtr = &c;
    int *iPtr = &j;
    printf("Address of double: %p \t Size of double: %ld\n", &i, sizeof(*dPtr));
    printf("Address of char: %p \t Size of char: %ld\n", &c, sizeof(*chPtr));
    printf("Address of int: %p \t Size of int: %ld\n", &j, sizeof(*iPtr));
    return 0;
}