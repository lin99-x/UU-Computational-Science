//
//  Task1_dowhile.c
//  
//
//  Created by Jinglin Gao on 2023-01-19.
//

#include <stdio.h>

int main() {
    int i = 100;
    do {
        printf("%d \t", i);
        i -= 4;
    }
    while(i >= 0);
    return 0;
}
