//
//  main.c
//  t302
//
//  Created by Jinglin Gao on 2023-01-19.
//

#include <stdio.h>

int main() {
    float i, j, k;
    printf("Input three real numbers: ");
    scanf("%f%f%f", &i, &j, &k);
    if (i < 0)
        i = -1 * i;
    if (j < 0)
        j = -1 * j;
    if (k < 0)
        k = -1 * k;
    if (i >= j && i >= k)
        printf("The largest number is %f. \n", i);
    if (j >= i && j >= k)
        printf("The largest number is %f. \n", j);
    if (k >= i && k >= i)
        printf("The largest number is %f. \n", k);
    return 0;
}
