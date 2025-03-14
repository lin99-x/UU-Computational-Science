//
//  main.c
//  Task3
//
//  Created by Jinglin Gao on 2023-01-19.
//

#include <stdio.h>

int main() {
    int i, j, sum, pro;
    printf("Input two integers: ");
    scanf("%d%d", &i, &j);
    if(i%2==0 && j%2==0) {
        sum = i + j;
        printf("These are two even numbers, sum is %d. \n", sum);
    }
    else {
        pro = i * j;
        printf("These are not two even numbers, product is %d. \n", pro);
    }
    return 0;
}
