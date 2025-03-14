#include <stdio.h>
#include <math.h>

int main() {
    double i, j;
    printf("Input: ");
    scanf("%lf", &i);
    j = sqrt(i);
    if (j - (int)j == 0) {
        printf("The number is a perfect square.");
    }
    else {
        printf("The number is not a perfect square.");
    }
    return 0;
}