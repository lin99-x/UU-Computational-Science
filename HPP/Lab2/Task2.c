//
//  Task2.c
//  
//
//  Created by Jinglin Gao on 2023-01-19.
//

#include <stdio.h>

int main() {
    // insert code here...
    int i, j ,k ,l;
    printf("Input: ");
    scanf("%d%d", &i, &j);
    for(k=0; k<i; k++) {
        for(l=0; l<j; l++)
            printf("*");
        printf("\n");
    }
    return 0;
}
