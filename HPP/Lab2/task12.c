#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    int i,j,flag=0,k,l,m,n;
    int *arr;
    printf("Input a number: ");
    scanf("%d", &i);
    n = i;
    arr = (int*)malloc(i*sizeof(int));
    printf("Input %d numbers for an array: ", i);
    for(j=0; j<i; j++) {
        scanf("%d", &arr[j]);
    }
    for(j=0; j<i; j++) {
        for(l=2; l<=sqrt(arr[j]); l++) {
            if(arr[j]%l == 0) {
                //is not prime number
                flag = 1;
                break;
            }
        }
        if(arr[j] <= 1) {
            flag = 1;
        }
        if(flag == 0) {
            n -= 1;
            for(m=j; m<i; m++) {
                arr[m] = arr[m+1];
            }
            j -= 1;
        }
        flag = 0;
    }
    arr = realloc(arr, n*sizeof(int));
    printf("n is: %d\n",n);
    for(j=0; j<n; j++) {
        printf("%d\t", arr[j]);
    }
    return 0;
}