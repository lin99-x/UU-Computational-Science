#include <stdio.h>

int main() {
    int i, inverse, remainder, orgi;
    printf("Input: ");
    scanf("%d", &i);
    inverse = 0;
    orgi = i;
    while (i != 0) {
        remainder =  i % 10;
        inverse = inverse * 10 + remainder;
        i = i / 10;
    }
    if (inverse == orgi) {
        printf("Output: it is a parlindrome");
        printf("%d", inverse);
    }
    else {
        printf("Output: it is not a palindrom");
        printf("%d", inverse);
    }
    return 0;
}