#include <stdio.h>

void print_int_1(int x) {
    printf("Here is the number: %d\n", x);
}

void print_int_2(int x) {
    printf("Wow, %d is really an impressive number!\n", x);
}

int main() {
    int i=8;
    void (*foo)(int);
    foo = &print_int_2;
    foo(i);
    return 0;
}

