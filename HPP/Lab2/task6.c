#include <stdio.h>

int main() {
    int i, j, k ,l;
    printf("Enter dividend: ");
    scanf("%d", &i);
    printf("Enter divisor: ");
    scanf("%d", &j);
    // quotient
    k = i / j;
    // remainder
    l = i % j;
    printf("Quotient: %d\n", k);
    printf("Remainder: %d\n", l);
    return 0;
}