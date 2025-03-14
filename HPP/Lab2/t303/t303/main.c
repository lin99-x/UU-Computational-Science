//
//  main.c
//  t303
//
//  Created by Jinglin Gao on 2023-01-19.
//

#include <stdio.h>

int main() {
    // insert code here...
    float i, j, k;
    printf("Input three real numbers: ");
    scanf("%f%f%f", &i, &j, &k);
    if (i < 0)
        i = -i;
    if (j < 0)
        j = -j;
    if (k < 0)
        k = -k;
    if ((i >= j && i <= k) || (i >= k && i <= j))
        printf("The second largest number is %f. \n", i);
    if ((j >= i && j <= k) || (j >= k && j <= i))
        printf("The second largest number is %f. \n", j);
    if ((k >= i && k <= j) || (k >= j && k <= i))
        printf("The second largest number is %f. \n", j);
    return 0;
}
