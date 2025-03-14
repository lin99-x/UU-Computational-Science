#include <stdio.h>
#include <stdlib.h>


int main(int argc, char const *argv[]) {
    char row;
    int result;
    if (argc == 2) {
        row = atoi(argv[1]);
    }
    else if (argc > 2) {
        printf("Too much parameters given.");
    }
   // printf("Row is: %d", row);
    
    for (int i=1; i<=row; i++) {
        result = 1;
        for (int j=1; j<i; j++) {
            printf("%d\t", result);
            result = result * (i - j)/j;
        }
        printf("%d", result);
        printf("\n");
    }

    return 0;
}