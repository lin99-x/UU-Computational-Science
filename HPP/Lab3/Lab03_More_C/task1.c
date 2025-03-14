#include <stdio.h>

int main() {
    int matrix[5][5] = {
        {0, 1, 1, 1, 1},
        {-1, 0, 1, 1, 1},
        {-1, -1, 0, 1, 1},
        {-1, -1, -1, 0, 1},
        {-1, -1, -1, -1, 0}
    };
    int i, j;
    for(i=0; i<5; i++) {
        for(j=0; j<5; j++) {
            printf("%d\t", matrix[i][j]);
            if(j==4) {
                printf("\n");
            }
        }
    }
    return 0;
}