#include <stdio.h>
#include <stdlib.h>

int CmpDouble(const void * a, const void * b) {
    // printf("Now the following two values are compared %.1lf & %.1lf", *(double*)a, *(double*)b);
    return (*(double*)a - *(double*)b);
}

int main() {
    double arrDouble[] = {9.3, -2.3, 1.2, -0.4, 2, 9.2, 1, 2.1, 0, -9.2};
    int arrlen = 10;
    qsort(arrDouble, arrlen, sizeof(double), CmpDouble);
    for(int i=0; i<arrlen; i++) {
        printf("%.1lf ", arrDouble[i]);
    }
    return 0;
}