#include <stdio.h>
#include <math.h>

int main() {
    float a = 100.999;
    int i, j;
    double r;
    for (i = 1; i <30; i++) {
        a = a * 100;
        printf("Now is the %dth loop\n", i);
        printf("Now the value of a is: %f\n", a);
    }
    // do something with inf
    j = 1;
    a = j + a;
    printf("Add 1 to inf: %f\n", a);

    j = -4;
    r = sqrt(j);
    printf("Sqrt -4 is: %lf\n", r);
    r = r + 1;
    printf("Add 1 to nan: %lf\n", r);
    return 0;
}