#include <stdio.h>
#include <stdlib.h>

int main() {
    int *arr;
    int current_size=10, i=0, sum=0;
    arr = (int *)malloc(current_size*sizeof(int));
    printf("Input: ");
    while(scanf("%d", &arr[i])!=0) {
        //scanf("%d", &arr[i]);
        i++;
        if(i>=current_size) {
            current_size += 10;
            arr = realloc(arr, current_size*sizeof(int));
        }
    }
    printf("Output: \n");
    for(int k=0; k<i; k++) {
        sum += arr[k];
        printf("%d\t",arr[k]);
    }
    printf("\nSum: %d", sum);
    return 0;
}